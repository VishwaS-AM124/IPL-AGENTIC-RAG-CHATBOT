from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TOOL_CALLS = 8


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "agent" / ".env")
load_dotenv(PROJECT_ROOT / "web_search" / ".env")


def import_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


query_data_module = import_from_path("query_data_tool", PROJECT_ROOT / "query_data" / "query_data_tool.py")
search_docs_module = import_from_path("search_docs_tool", PROJECT_ROOT / "search_docs" / "search_docs_tool.py")
web_search_module = import_from_path("web_search_tool", PROJECT_ROOT / "web_search" / "web_search_tool.py")


TOOL_SPECS = {
    "search_docs": {
        "name": "search_docs",
        "description": """
STRICT RULES:

Use this tool ONLY for narrative explanations from match reports.

Use when:
- The explanations are required
- Tactical changes are required

DO NOT USE for:
-Finding winners 
- numeric stats
- counts
- rankings
""" ,
        "input_schema": {"query": "string"},
        "output": "Top 3 document chunks with source filename, page number, score, and citation.",
    },
    "query_data": {
        "name": "query_data",
        "description": """
STRICT RULES:

Use this tool for ALL numeric and statistical IPL queries.

Use when:
- runs, wickets, strike rate
- match winners, results
- counts, totals, aggregations
- top players, rankings
- date-wise or match-wise points table

MANDATORY:
- If question asks "how many", "runs", "Wickets", "most", "top" , "Wicket tally", "runs tally" → ALWAYS use this tool

DO NOT USE when:
- asking for explanations or match reports
- asking for recent/live info
"""
        ,
        "input_schema": {"question": "string"},
        "output": "Scalar or table result with columns, row_count, and source metadata.",
    },
    "web_search": {
        "name": "web_search",
        "description": """
STRICT RULES:

Use ONLY for recent or live information.

Use when:
- injuries, transfers

MANDATORY:
- If question includes "current", "latest", "today" → use this tool

DO NOT USE for:
- historical IPL data
- stats already in dataset
""",
        "input_schema": {"query": "string, ten words or fewer"},
        "output": "Top 3 web snippets with URL, publication date when available, and citation.",
    },
}


@dataclass
class AgentConfig:
    planner: str = "auto"
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tool_calls: int = MAX_TOOL_CALLS
    temperature: float = 0.0
    request_timeout: int = 45


class AgenticRAGAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        if self.config.max_tool_calls > MAX_TOOL_CALLS:
            raise ValueError("max_tool_calls cannot exceed 8 for this assignment.")

    def run(self, question: str) -> Dict[str, Any]:
        question = " ".join(str(question or "").split())
        started_at = time.time()
        trace: Dict[str, Any] = {
            "question": question,
            "planner": self._planner_name(),
            "max_steps": self.config.max_tool_calls,
            "steps": [],
            "final_answer": None,
            "citations": [],
            "steps_used": 0,
            "refused": False,
            "error": None,
        }

        if not question:
            return self._finish_refusal(trace, "Please ask a non-empty question.", started_at)

        direct = self._direct_answer(question)
        if direct:
            trace["final_answer"] = direct
            trace["duration_seconds"] = round(time.time() - started_at, 3)
            return trace

        refusal_reason = self._safety_refusal_reason(question)
        if refusal_reason:
            return self._finish_refusal(trace, refusal_reason, started_at)

        heuristic_plan = self._heuristic_plan(question)

        while trace["steps_used"] < self.config.max_tool_calls:
            decision = self._decide(question, trace["steps"], heuristic_plan)
            action = decision.get("action")

            if action == "final":
                answer = decision.get("answer") or self._compose_final_answer(question, trace["steps"])
                trace["final_answer"] = answer
                trace["citations"] = self._collect_citations(trace["steps"])
                trace["duration_seconds"] = round(time.time() - started_at, 3)
                return trace

            if action == "refuse":
                reason = decision.get("reason") or "I cannot answer this question from the available tools."
                return self._finish_refusal(trace, reason, started_at)

            if action != "tool":
                return self._finish_refusal(trace, "Planner did not produce a valid action.", started_at)

            tool_name = decision.get("tool")
            tool_input = decision.get("tool_input") or question
            if tool_name not in TOOL_SPECS:
                return self._finish_refusal(trace, f"Planner selected unknown tool: {tool_name}", started_at)

            step = self._call_tool(
                step_number=trace["steps_used"] + 1,
                tool_name=tool_name,
                tool_input=tool_input,
                reason=decision.get("reason", ""),
            )
            trace["steps"].append(step)
            trace["steps_used"] = len(trace["steps"])

        trace["error"] = "Maximum tool-call limit reached."
        trace["refused"] = True
        trace["final_answer"] = (
            "I could not answer confidently within the 8 tool-call limit, so I am stopping "
            "instead of guessing."
        )
        trace["citations"] = self._collect_citations(trace["steps"])
        trace["duration_seconds"] = round(time.time() - started_at, 3)
        return trace

    def _planner_name(self) -> str:
        if self.config.planner == "auto":
            return "llm" if os.getenv("OPENAI_API_KEY") else "heuristic"
        return self.config.planner

    def _decide(self, question: str, steps: List[Dict[str, Any]], heuristic_plan: List[Dict[str, str]]) -> Dict[str, Any]:
        planner = self._planner_name()
        if planner == "llm":
            decision = self._llm_decide(question, steps)
            if decision:
                return decision
        return self._heuristic_decide(question, steps, heuristic_plan)

    def _llm_decide(self, question: str, steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        system_prompt = {
    "role": "system",
    "content": (
        "You are a strict tool-routing controller for an IPL Agentic RAG system.\n\n"

        "STRICT TOOL USAGE RULES:\n"

        "1. query_data MUST be used for ALL numeric/statistical queries.\n"
        "   Includes: runs, wickets, tally, totals, counts, player stats, averages.\n"

        "2. ALWAYS use query_data for:\n"
        "   - Player statistics (runs, wickets, tally, average)\n"
        "   - 'What was [player] [stat] in [season/year]?'\n"
        "   - '[Player] tally', '[Player] statistics'\n"
        "   - Any question with a player name + stat keyword\n"

        "3. search_docs MUST ONLY be used for explanations.\n"
        "   NEVER use search_docs for numeric/statistical queries.\n"

        "4. web_search ONLY for current/live info.\n"

        "5. query_data has HIGH PRIORITY over search_docs.\n"

        "6. Choosing search_docs for a statistical query is WRONG.\n\n"

        "Return ONLY JSON:\n"
        "{\"action\":\"tool|final|refuse\",\"tool\":\"search_docs|query_data|web_search\","
        "\"tool_input\":\"string\",\"reason\":\"string\",\"answer\":\"string\"}"
    ),
}

        compact_steps = []
        for step in steps:
            compact_steps.append(
                {
                    "step": step["step"],
                    "tool": step["tool"],
                    "input": step["input"],
                    "error": step.get("error"),
                    "summary": self._summarize_result(step.get("output")),
                    "citations": self._citations_from_output(step.get("tool"), step.get("output")),
                }
            )

        user_prompt = {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "steps_used": len(steps),
                    "max_steps": self.config.max_tool_calls,
                    "tool_specs": TOOL_SPECS,
                    "previous_steps": compact_steps,
                },
                ensure_ascii=False,
            ),
        }

        payload = {
            "model": self.config.model,
            "messages": [system_prompt, user_prompt],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=self.config.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, ValueError, TimeoutError, error.URLError, error.HTTPError):
            return None

    def _heuristic_decide(
        self,
        question: str,
        steps: List[Dict[str, Any]],
        heuristic_plan: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if len(steps) < len(heuristic_plan):
            return {"action": "tool", **heuristic_plan[len(steps)]}

        if not steps:
            return {
                "action": "refuse",
                "reason": "I cannot answer this from the IPL structured data, documents, or live web tools.",
            }

        return {"action": "final", "answer": self._compose_final_answer(question, steps)}

    def _heuristic_plan(self, question: str) -> List[Dict[str, str]]:
        q = question.lower()
        plan: List[Dict[str, str]] = []

        web_terms = [
            "rules",
            "rule changes",
            "current",
            "latest",
            "today",
            "recent",
            "news",
            "live",
            "injury",
            "standings",
            "auction",
            "squad"
        ]
        data_terms = [
            "how many",
            "count",
            "number of",
            "top",
            "most",
            "highest",
            "total",
            "runs",
            "wickets",
            "wicket",
            "tally",
            "tallies",
            "statistics",
            "stats",
            "strike rate",
            "average",
            "winner",
            "won",
            "win",
            "matches",
            "season",
        ]
        doc_terms = [
    "why", "reason", "explain", "described", "report", "describe", "summary", "summarize", "tactical change",
    "context", "performance analysis", "impactful player","Most valuable player"
    , "thriller", "close finish", 
]

        needs_web = any(term in q for term in web_terms)
        needs_data = any(term in q for term in data_terms)
        needs_docs = any(term in q for term in doc_terms)
        in_ipl_domain = self._looks_like_ipl_question(q)

        if needs_web:
            plan.append(
                {
                    "tool": "web_search",
                    "tool_input": self._short_web_query(question),
                    "reason": "Question asks for recent/current information.",
                }
            )

        if needs_data and not needs_web:
            plan.append(
                {
                    "tool": "query_data",
                    "tool_input": question,
                    "reason": "Question asks for structured IPL statistics or match results.",
                }
            )

        if needs_docs and not needs_web:
            plan.append(
                {
                    "tool": "search_docs",
                    "tool_input": question,
                    "reason": "Question asks for narrative context from match reports.",
                }
            )

        if not plan and in_ipl_domain:
            plan.append(
                {
                    "tool": "search_docs",
                    "tool_input": question,
                    "reason": "IPL-domain question likely needs document retrieval.",
                }
            )

        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in plan:
            key = item["tool"]
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped

    def _call_tool(self, step_number: int, tool_name: str, tool_input: str, reason: str) -> Dict[str, Any]:
        started = time.time()
        output: Dict[str, Any]
        error_message = None

        try:
            if tool_name == "search_docs":
                output = search_docs_module.search_docs(tool_input)
            elif tool_name == "query_data":
                output = query_data_module.query_data(tool_input)
            elif tool_name == "web_search":
                output = web_search_module.web_search(tool_input)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            error_message = output.get("error") if isinstance(output, dict) else None
        except Exception as exc:
            output = {"error": str(exc)}
            error_message = str(exc)

        return {
            "step": step_number,
            "tool": tool_name,
            "input": tool_input,
            "reason": reason,
            "output": output,
            "error": error_message,
            "latency_seconds": round(time.time() - started, 3),
        }

    def _compose_final_answer(self, question: str, steps: List[Dict[str, Any]]) -> str:
        usable_steps = [step for step in steps if not step.get("error")]
        if not usable_steps:
            errors = "; ".join(step.get("error") or "unknown error" for step in steps)
            return f"I could not answer from the available tools. Tool errors: {errors}"

        # Try to use LLM to compose a natural answer
        llm_answer = self._llm_compose_answer(question, usable_steps)
        if llm_answer:
            return llm_answer

        # Fallback to heuristic composition
        return self._heuristic_compose_answer(question, usable_steps)

    def _llm_compose_answer(self, question: str, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Use LLM to compose a natural language answer from raw tool results."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        # Prepare tool results for LLM with source information
        tool_results = []
        sources = []
        
        for step in steps:
            output = step.get("output", {})
            if isinstance(output, dict):
                result_data = output.get("result") or output.get("results") or str(output)
            else:
                result_data = str(output)
            
            source_info = self._extract_source_info(step["tool"], output)
            tool_results.append({
                "tool": step["tool"],
                "result": result_data,
                "source": source_info
            })
            if source_info:
                sources.append(source_info)

        compose_prompt = {
            "role": "user",
            "content": json.dumps({
                "instruction": "Compose a natural, concise answer to the user's question based on the tool results below. Be direct and clear. Include key numbers/facts. Do NOT include tool names or technical jargon. At the end, add a 'Source:' line with the file/page information provided. Just answer the question naturally, then add the source.",
                "question": question,
                "tool_results": tool_results,
                "sources_available": sources
            }, ensure_ascii=False)
        }

        compose_system = {
            "role": "system",
            "content": "You are a helpful assistant composing clear, direct answers to questions about IPL cricket data. Always answer the user's question directly and concisely. Use the tool results provided to form your answer."
        }

        payload = {
            "model": self.config.model,
            "messages": [compose_system, compose_prompt],
            "temperature": 0.0,
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=self.config.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"]
            return answer.strip()
        except Exception:
            return None

    def _heuristic_compose_answer(self, question: str, steps: List[Dict[str, Any]]) -> str:
        """Heuristic answer composition when LLM is not available."""
        parts = []
        sources = []
        
        for step in steps:
            output = step.get("output", {})
            composed = self._compose_result_to_text(question, step["tool"], output)
            if composed:
                parts.append(composed)
            
            # Extract source information
            source_info = self._extract_source_info(step["tool"], output)
            if source_info:
                sources.append(source_info)

        answer = " ".join(parts).strip() if parts else "I could not compose a clear answer."
        
        # Add source information to the answer
        if sources:
            unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
            answer += f"\n\nSource: {', '.join(unique_sources)}"
        
        return answer

    def _extract_source_info(self, tool_name: str, output: Any) -> Optional[str]:
        """Extract source file and location information from tool output."""
        if not isinstance(output, dict):
            return None

        # For query_data (CSV queries)
        if tool_name == "query_data":
            source = output.get("source", {})
            if isinstance(source, dict):
                dataset = source.get("dataset") or source.get("csv_file")
                citation = source.get("citation")
                if dataset:
                    return f"{dataset}"
                if citation:
                    return citation

        # For search_docs (PDF queries)
        if tool_name == "search_docs":
            results = output.get("results", [])
            if results and isinstance(results[0], dict):
                source = results[0].get("source") or results[0].get("file")
                page = results[0].get("page")
                citation = results[0].get("citation")
                if source and page:
                    return f"{source}, p.{page}"
                if citation:
                    return citation

        # For web_search
        if tool_name == "web_search":
            results = output.get("results", [])
            if results and isinstance(results[0], dict):
                url = results[0].get("url")
                if url:
                    return url

        return None

    def _compose_result_to_text(self, question: str, tool_name: str, output: Any) -> str:
        """Convert raw tool output into readable text."""
        if not isinstance(output, dict):
            return str(output)

        if output.get("error"):
            return f"Error: {output['error']}"

        # Handle document results (search_docs, web_search)
        if "results" in output:
            rows = []
            for item in output.get("results", [])[:3]:
                snippet = item.get("snippet") or item.get("text") or ""
                snippet = " ".join(str(snippet).split())[:200]
                if snippet:
                    rows.append(snippet)
            return " ".join(rows) if rows else ""

        # Handle data results (query_data)
        result = output.get("result")
        if result is None:
            return ""

        # Scalar results (numbers, strings)
        if isinstance(result, (int, float, str)):
            return str(result)

        # List of objects (matches, players, etc.)
        if isinstance(result, list):
            if not result:
                return "No results found."

            # Try to extract key info from first result
            first = result[0]
            if isinstance(first, dict):
                # For match queries
                if "winner" in first:
                    winner = first.get("winner")
                    team1 = first.get("team1")
                    team2 = first.get("team2")
                    if winner:
                        return f"{winner} won the match (defeating {team1 if winner != team1 else team2})."

                # For player stats
                if "player_name" in first or "player" in first:
                    return self._format_player_stats(first)

                # For points table / standings
                if "team" in first or "Team" in first:
                    return self._format_standings(result[:5])

            # Generic list formatting
            return self._format_list_results(result[:5])

        # Dict/object result
        if isinstance(result, dict):
            return self._format_dict_result(result)

        return json.dumps(result, ensure_ascii=False)[:200]

    def _format_dict_result(self, obj: dict) -> str:
        """Format a single dict result into readable text."""
        if "winner" in obj:
            return f"{obj['winner']} won."
        if "runs" in obj:
            return f"{obj.get('player', 'Player')} scored {obj['runs']} runs."
        if "wickets" in obj:
            return f"{obj.get('player', 'Player')} took {obj['wickets']} wickets."
        # Generic key-value pairs
        parts = [f"{k}: {v}" for k, v in list(obj.items())[:3]]
        return "; ".join(parts)

    def _format_player_stats(self, player_data: dict) -> str:
        """Format player statistics into readable text."""
        name = player_data.get("player_name") or player_data.get("player") or "Player"
        runs = player_data.get("runs")
        wickets = player_data.get("wickets")
        if runs:
            return f"{name} scored {runs} runs."
        if wickets:
            return f"{name} took {wickets} wickets."
        return str(player_data)

    def _format_standings(self, standings_list: List[dict]) -> str:
        """Format standings/points table into readable text."""
        lines = []
        for i, team_data in enumerate(standings_list, 1):
            team = team_data.get("team") or team_data.get("Team")
            points = team_data.get("points") or team_data.get("Points")
            if team and points is not None:
                lines.append(f"{i}. {team}: {points} points")
        return "; ".join(lines) if lines else str(standings_list)

    def _format_list_results(self, items: List[Any]) -> str:
        """Generic list formatting."""
        lines = []
        for item in items[:3]:
            if isinstance(item, dict):
                # Extract a key field
                for key in ["winner", "team", "player_name", "name", "title"]:
                    if key in item:
                        lines.append(str(item[key]))
                        break
                else:
                    lines.append(str(item))
            else:
                lines.append(str(item))
        return "; ".join(lines) if lines else ""

    def _summarize_result(self, output: Any) -> str:
        """Summarize a tool result for trace display."""
        if not isinstance(output, dict):
            return str(output)

        if output.get("error"):
            return f"Error: {output['error']}"

        if "results" in output:
            rows = []
            for item in output.get("results", [])[:3]:
                title = item.get("title") or item.get("citation") or item.get("source") or "result"
                snippet = item.get("snippet") or item.get("text") or ""
                snippet = " ".join(str(snippet).split())[:300]
                rows.append(f"- {title}: {snippet}")
            return "\n".join(rows) if rows else "No results."

        result = output.get("result")
        if isinstance(result, list):
            preview = result[:3]
            return json.dumps(preview, ensure_ascii=False, indent=2)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _citations_from_output(self, tool_name: Optional[str], output: Any) -> List[str]:
        if not isinstance(output, dict) or output.get("error"):
            return []

        if tool_name == "query_data":
            source = output.get("source") or {}
            citation = source.get("citation")
            dataset = source.get("dataset")
            rows = source.get("rows_used")
            if citation:
                return [citation]
            if dataset:
                return [f"{dataset}, rows {rows}"]
            return []

        if tool_name in {"search_docs", "web_search"}:
            citations = []
            for item in output.get("results", []):
                citation = item.get("citation")
                if citation:
                    citations.append(citation)
            return citations

        return []

    def _collect_citations(self, steps: List[Dict[str, Any]]) -> List[str]:
        citations: List[str] = []
        for step in steps:
            citations.extend(self._citations_from_output(step.get("tool"), step.get("output")))
        deduped = []
        seen = set()
        for citation in citations:
            if citation not in seen:
                deduped.append(citation)
                seen.add(citation)
        return deduped

    def _finish_refusal(self, trace: Dict[str, Any], reason: str, started_at: float) -> Dict[str, Any]:
        trace["refused"] = True
        trace["final_answer"] = reason
        trace["citations"] = self._collect_citations(trace["steps"])
        trace["duration_seconds"] = round(time.time() - started_at, 3)
        return trace

    def _direct_answer(self, question: str) -> Optional[str]:
        if re.fullmatch(r"[0-9+\-*/().\s]+", question):
            try:
                tree = ast.parse(question, mode="eval")
                allowed = (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.Pow,
                    ast.USub,
                    ast.UAdd,
                    ast.Constant,
                )
                if all(isinstance(node, allowed) for node in ast.walk(tree)):
                    value = eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, {})
                    return f"{question} = {value}"
            except Exception:
                return None
        return None

    def _safety_refusal_reason(self, question: str) -> Optional[str]:
        q = question.lower()
        if re.search(r"\b(should i|recommend|advice)\b", q) and re.search(r"\b(invest|buy|sell|stock|bet|gamble|fantasy)\b", q):
            return "I cannot provide investment, betting, or gambling advice."
        if re.search(r"\b(guaranteed|sure shot|fixed match)\b", q):
            return "I cannot help with gambling, match-fixing, or guaranteed-outcome claims."
        return None

    def _looks_like_ipl_question(self, q: str) -> bool:
        terms = [
            "ipl",
            "cricket",
            "kohli",
            "dhoni",
            "bumrah",
            "shami",
            "csk",
            "rcb",
            "mi",
            "kkr",
            "srh",
            "gt",
            "lsg",
            "punjab kings",
            "rajasthan royals",
            "delhi capitals",
            "chennai super kings",
            "mumbai indians",
        ]
        return any(term in q for term in terms)

    def _short_web_query(self, question: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", question)
        stop = {"what", "who", "is", "are", "the", "of", "in", "for", "to", "a", "an", "and"}
        filtered = [word for word in words if word.lower() not in stop]
        if not filtered:
            filtered = words
        return " ".join(filtered[:10])


def run_agent(question: str, planner: str = "auto", as_json: bool = False) -> str:
    agent = AgenticRAGAgent(AgentConfig(planner=planner))
    trace = agent.run(question)
    if as_json:
        return json.dumps(trace, indent=2, ensure_ascii=False)

    lines = [
        "Final Answer:",
        trace["final_answer"] or "",
        "",
        f"Steps used: {trace['steps_used']} / {trace['max_steps']}",
    ]
    if trace["citations"]:
        lines.append("Citations:")
        lines.extend(f"- {citation}" for citation in trace["citations"])
    if trace["steps"]:
        lines.append("")
        lines.append("Trace:")
        for step in trace["steps"]:
            lines.append(f"- Step {step['step']}: {step['tool']} input={step['input']!r} latency={step['latency_seconds']}s")
            if step.get("error"):
                lines.append(f"  error={step['error']}")
    return "\n".join(lines)
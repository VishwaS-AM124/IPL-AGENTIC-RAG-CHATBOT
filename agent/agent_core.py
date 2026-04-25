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
        "description": search_docs_module.TOOL_DESCRIPTION,
        "input_schema": {"query": "string"},
        "output": "Top 3 document chunks with source filename, page number, score, and citation.",
    },
    "query_data": {
        "name": "query_data",
        "description": (
            "Query the local structured IPL datasets with safe pandas filters and aggregations. "
            "Use this for match winners, season summaries, team win counts, player runs, strike "
            "rates, wickets, top scorers, top wicket takers, and player profile fields. Do not "
            "use it for current news or narrative explanations from match reports."
        ),
        "input_schema": {"question": "string"},
        "output": "Scalar or table result with columns, row_count, and source metadata.",
    },
    "web_search": {
        "name": "web_search",
        "description": web_search_module.TOOL_DESCRIPTION,
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
                "You are the tool-routing controller for an IPL Agentic RAG system. "
                "Return one JSON object only. Choose one of these actions: "
                "tool, final, refuse. Use at most one tool per decision. "
                "Use search_docs for unstructured PDF match-report explanations. "
                "Use query_data for structured IPL stats, counts, winners, runs, wickets, strike rates, and tables. "
                "Use web_search only for recent/current/live information, and keep its query <= 10 words. "
                "Refuse investment, betting, unsafe, unrelated, or unanswerable questions. "
                "If tool results already answer the question, action=final with a cited answer. "
                "JSON schema: "
                "{\"action\":\"tool|final|refuse\",\"tool\":\"search_docs|query_data|web_search\","
                "\"tool_input\":\"string\",\"reason\":\"string\",\"answer\":\"string\"}."
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
            "current",
            "latest",
            "today",
            "recent",
            "news",
            "live",
            "injury",
            "standings",
            "auction",
            "squad",
            "2025",
            "2026",
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
            "strike rate",
            "winner",
            "won",
            "win",
            "matches",
            "season",
        ]
        doc_terms = [
            "why",
            "reason",
            "explain",
            "described",
            "report",
            "context",
            "performance",
            "impact player",
            "rule",
            "thriller",
            "close finish",
            "spell",
            "batting",
            "bowling",
            "final",
            "qualifier",
            "eliminator",
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

        parts = [f"Question: {question}", ""]
        for step in usable_steps:
            parts.append(f"Step {step['step']} used `{step['tool']}`:")
            parts.append(self._summarize_result(step["output"]))
            citations = self._citations_from_output(step["tool"], step["output"])
            if citations:
                parts.append("Citations: " + "; ".join(citations[:5]))
            parts.append("")
        parts.append(f"Steps used: {len(steps)} / {self.config.max_tool_calls} max")
        return "\n".join(parts).strip()

    def _summarize_result(self, output: Any) -> str:
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
            preview = result[:5]
            return json.dumps(preview, ensure_ascii=False, indent=2)
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

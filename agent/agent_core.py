from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TOOL_CALLS = 8

# Score below which search_docs results are considered off-topic.
# FAISS cosine similarity + metadata boost: relevant chunks score ~0.40-0.75,
# irrelevant chunks score ~0.20-0.34.  Paste the output of the diagnostic command
# to tune this if needed.
RELEVANCE_THRESHOLD = 0.35


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


query_data_module = import_from_path("query_data_tool",  PROJECT_ROOT / "query_data"  / "query_data_tool.py")
search_docs_module = import_from_path("search_docs_tool", PROJECT_ROOT / "search_docs" / "search_docs_tool.py")
web_search_module  = import_from_path("web_search_tool",  PROJECT_ROOT / "web_search"  / "web_search_tool.py")


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
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_tool_calls: int = MAX_TOOL_CALLS
    temperature: float = 0.0
    request_timeout: int = 45


# Groq is OpenAI-compatible — same JSON schema, different base URL and key name.
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_api_key() -> str:
    """Return the active LLM API key (Groq preferred, OpenAI as fallback)."""
    return os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _get_base_url() -> str:
    """Return the chat completions endpoint for whichever provider is configured."""
    if os.getenv("GROQ_API_KEY"):
        return GROQ_BASE_URL
    return "https://api.openai.com/v1/chat/completions"


class AgenticRAGAgent:

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        if self.config.max_tool_calls > MAX_TOOL_CALLS:
            raise ValueError("max_tool_calls cannot exceed 8 for this assignment.")

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, question: str) -> Dict[str, Any]:
        question = " ".join(str(question or "").split())
        started_at = time.time()
        trace: Dict[str, Any] = {
            "question":     question,
            "planner":      self._planner_name(),
            "max_steps":    self.config.max_tool_calls,
            "steps":        [],
            "final_answer": None,
            "citations":    [],
            "steps_used":   0,
            "refused":      False,
            "error":        None,
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

        # Check for false premises before proceeding
        premise_correction = self._verify_premise(question)
        if premise_correction:
            trace["final_answer"] = premise_correction
            trace["duration_seconds"] = round(time.time() - started_at, 3)
            return trace

        heuristic_plan = self._heuristic_plan(question)

        while trace["steps_used"] < self.config.max_tool_calls:
            decision = self._decide(question, trace["steps"], heuristic_plan)
            action   = decision.get("action")

            if action == "final":
                answer = decision.get("answer") or self._compose_final_answer(question, trace["steps"])
                trace["final_answer"] = answer
                trace["citations"]    = self._collect_citations(trace["steps"])
                trace["duration_seconds"] = round(time.time() - started_at, 3)
                return trace

            if action == "refuse":
                reason = decision.get("reason") or "I cannot answer this question from the available tools."
                return self._finish_refusal(trace, reason, started_at)

            if action != "tool":
                return self._finish_refusal(trace, "Planner did not produce a valid action.", started_at)

            tool_name  = decision.get("tool")
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

        trace["error"]        = "Maximum tool-call limit reached."
        trace["refused"]      = True
        trace["final_answer"] = (
            "I could not answer confidently within the 8 tool-call limit, so I am stopping "
            "instead of guessing."
        )
        trace["citations"]    = self._collect_citations(trace["steps"])
        trace["duration_seconds"] = round(time.time() - started_at, 3)
        return trace

    # ── planner dispatch ──────────────────────────────────────────────────────

    def _planner_name(self) -> str:
        if self.config.planner == "auto":
            return "llm" if _get_api_key() else "heuristic"
        return self.config.planner

    def _decide(
        self,
        question: str,
        steps: List[Dict[str, Any]],
        heuristic_plan: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        planner = self._planner_name()
        if planner == "llm":
            decision = self._llm_decide(question, steps)
            if decision:
                return decision
        return self._heuristic_decide(question, steps, heuristic_plan)

    # ── LLM planner ───────────────────────────────────────────────────────────

    def _llm_decide(self, question: str, steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        api_key = _get_api_key()
        if not api_key:
            return None
        base_url = _get_base_url()

        system_prompt = {
            "role": "system",
            "content": (
                "You are the tool-routing controller AND answer synthesizer for an IPL Agentic RAG system. "
                "Return exactly one JSON object. Always choose one of: action=tool, action=final, action=refuse.\n\n"

                "TOOL ROUTING RULES:\n"
                "- search_docs: Use for narrative explanations, match context, player performance "
                "descriptions, tactical analysis, or any fact written in the PDF match reports. "
                "CRITICAL: Use for 'How did X win/lose?' (asking for manner/tactics/narrative). "
                "Only use if the PDF corpus is likely to contain the answer.\n"
                "- query_data: Use for structured stats — match winners, runs, wickets, strike "
                "rates, win counts, season summaries, head-to-head records, player profiles. "
                "Use for 'Who won?' but NOT for 'How did they win?' (that's narrative). "
                "Also use to verify factual premises about specific match results.\n"
                "- web_search: Use for current/live/recent information (2025 news, injuries, "
                "live scores, rule announcements, regulatory changes, NEW RULES, RULE CHANGES). "
                "Query <= 10 words.\n\n"

                "CRITICAL DISTINCTION:\n"
                "- 'How did CSK win the 2023 final?' → search_docs (narrative: tactics, key moments)\n"
                "- 'Who won the 2023 final?' → query_data (fact: winner name)\n"
                "- 'How many runs did Kohli score?' → query_data (numeric stat)\n"
                "- 'Why did MI lose?' → search_docs (narrative: reasons, analysis)\n\n"

                "SPECIAL CASES:\n"
                "1. RULE CHANGES: Any question about 'new rule', 'rule change', 'impact of rule' "
                "should use web_search ONLY, never search_docs. Rule changes are official "
                "announcements found online, not in old match reports.\n"
                "2. OFF-TOPIC SEARCH_DOCS: If search_docs returns irrelevant chunks (relevance=false "
                "or max_score < 0.35), do NOT answer from that data. "
                "Use web_search as fallback or refuse if question can't be answered.\n\n"

                "FACTUAL PREMISE VERIFICATION RULE (CRITICAL - CHECK FIRST):\n"
                "ALWAYS check the question for embedded factual claims about specific match outcomes:\n"
                "  - 'Why did [TEAM] lose the [YEAR] final?' → Premise: [TEAM] played the final\n"
                "  - 'How did [TEAM] win [EVENT]?' → Premise: [TEAM] won that event\n"
                "  - 'Describe the performance in [SPECIFIC_MATCH]' → Premise: match happened\n\n"

                "IF FIRST STEP WAS PREMISE VERIFICATION (tool=query_data, reason contains 'premise'):\n"
                "  AFTER reading the results:\n"
                "  - If query_data found matching results (no error, data returned): "
                "The premise is TRUE. Proceed to answer using search_docs/web_search.\n"
                "  - If query_data returned ERROR or NO RESULTS: "
                "The premise is FALSE or unverifiable. Set action=final and CORRECT the user. "
                "Example: 'MI did not play the 2023 IPL final. The final was between CSK and GT.'\n"
                "  - NEVER proceed to search_docs after a failed premise check.\n\n"

                "IF NO PREMISE VERIFICATION DONE YET, AND question has embedded premise:\n"
                "  Recommend action=tool, tool=query_data, with reason='Verifying factual premise...'\n"
                "  Ask query_data: 'Did [TEAM] play in the [YEAR] final?' or 'Who won [EVENT]?'\n\n"

                "REFUSE RULES:\n"
                "Refuse investment advice, betting tips, match-fixing, guaranteed outcomes, "
                "and questions completely unrelated to IPL cricket.\n\n"

                "SYNTHESIZING THE FINAL ANSWER (CRITICAL):\n"
                "When you set action=final AND previous_steps contain tool results, you MUST:\n"
                "  1. Read the full tool output in previous_steps carefully.\n"
                "  2. Write 2-5 sentences of fluent English prose in the 'answer' field.\n"
                "  3. Extract and interpret the key facts — do NOT paste raw JSON or chunk text.\n"
                "  4. Cite each fact inline, e.g. (iplmatches.csv) or (IPL_2023&2024_merged.pdf, p.47).\n"
                "  5. If data from multiple tools was retrieved, integrate it into one coherent answer.\n"
                "  6. Never guess or hallucinate — if evidence is insufficient, say so clearly.\n\n"

                "JSON schema (all fields required, use empty string if not applicable):\n"
                "{\"action\":\"tool|final|refuse\","
                "\"tool\":\"search_docs|query_data|web_search\","
                "\"tool_input\":\"string\","
                "\"reason\":\"string\","
                "\"answer\":\"string (MUST be a fluent synthesized answer when action=final and evidence was retrieved)\"}"
            ),
        }

        # Build step summaries with FULL structured output so LLM can synthesize the answer
        compact_steps = []
        for step in steps:
            output = step.get("output", {})
            tool   = step.get("tool", "")

            # Pass the actual result data (not just a text summary) so LLM can write the answer
            full_result: Any = None
            if isinstance(output, dict) and not output.get("error"):
                if tool == "query_data":
                    res = output.get("result")
                    full_result = res[:10] if isinstance(res, list) else res
                elif tool == "search_docs":
                    full_result = [
                        {
                            "rank":     r.get("rank"),
                            "score":    r.get("score"),
                            "citation": r.get("citation"),
                            "text":     r.get("text", "")[:500],
                        }
                        for r in output.get("results", [])[:3]
                    ]
                elif tool == "web_search":
                    full_result = [
                        {"url": r.get("url"), "snippet": r.get("snippet", "")[:400]}
                        for r in output.get("results", [])[:3]
                    ]

            compact_steps.append({
                "step":        step["step"],
                "tool":        tool,
                "input":       step["input"],
                "error":       step.get("error"),
                "full_result": full_result,
                "citations":   self._citations_from_output(tool, output),
                "relevance":   self._relevance_summary(tool, output),
            })

        user_prompt = {
            "role": "user",
            "content": json.dumps(
                {
                    "question":       question,
                    "steps_used":     len(steps),
                    "max_steps":      self.config.max_tool_calls,
                    "tool_specs":     TOOL_SPECS,
                    "previous_steps": compact_steps,
                    "instruction":    (
                        "If you choose action=final, write a complete synthesized answer "
                        "in the 'answer' field based on the full_result data in previous_steps."
                    ),
                },
                ensure_ascii=False,
            ),
        }

        payload = {
            "model":           self.config.model,
            "messages":        [system_prompt, user_prompt],
            "temperature":     self.config.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens":      800,
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req  = request.Request(
                base_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                    "User-Agent":    "python-groq/0.9.0",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=self.config.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None

    # ── heuristic planner ─────────────────────────────────────────────────────

    def _heuristic_decide(
        self,
        question: str,
        steps: List[Dict[str, Any]],
        heuristic_plan: List[Dict[str, str]],
    ) -> Dict[str, Any]:

        already_used = {s["tool"] for s in steps}

        # Check last step for relevance — apply fallbacks before running next plan step
        if steps:
            last = steps[-1]
            relevance = self._relevance_summary(last.get("tool"), last.get("output"))

            # search_docs returned irrelevant chunks → fall back to web_search
            if (
                last.get("tool") == "search_docs"
                and not relevance["relevant"]
                and "web_search" not in already_used
            ):
                return {
                    "action":     "tool",
                    "tool":       "web_search",
                    "tool_input": self._short_web_query(question),
                    "reason":     (
                        f"search_docs returned low-relevance results "
                        f"(max score {relevance.get('max_score', 0):.3f} < {RELEVANCE_THRESHOLD}). "
                        "Falling back to web_search."
                    ),
                }

            # query_data returned no rows → fall back to search_docs
            if (
                last.get("tool") == "query_data"
                and not relevance["relevant"]
                and "search_docs" not in already_used
            ):
                return {
                    "action":     "tool",
                    "tool":       "search_docs",
                    "tool_input": question,
                    "reason":     "query_data returned no results. Trying search_docs.",
                }

        # Execute next step from the pre-built heuristic plan
        if len(steps) < len(heuristic_plan):
            return {"action": "tool", **heuristic_plan[len(steps)]}

        if not steps:
            return {
                "action": "refuse",
                "reason": "I cannot answer this from the IPL structured data, documents, or live web tools.",
            }

        return {"action": "final", "answer": self._compose_final_answer(question, steps)}

    def _heuristic_plan(self, question: str) -> List[Dict[str, str]]:
        q    = question.lower()
        plan: List[Dict[str, str]] = []

        # Rule changes and official announcements are current information (web-only)
        web_terms  = ["current", "latest", "today", "recent", "news", "live",
                      "injury", "standings", "auction", "squad", "2025", "2026",
                      "rule change", "new rule", "impact of rule", "official rule"]
        
        data_terms = ["how many", "count", "number of", "top", "most", "highest",
                      "total", "runs", "wickets", "strike rate", "winner", "won",
                      "win", "matches", "season"]
        
        # Remove "rule" from doc_terms to avoid confusion with rule changes
        doc_terms  = ["why", "reason", "explain", "described", "report", "context",
                      "performance", "impact player", "thriller", "close finish",
                      "spell", "batting", "bowling", "tactical", "strategy"]

        needs_web  = any(t in q for t in web_terms)
        needs_data = any(t in q for t in data_terms)
        needs_docs = any(t in q for t in doc_terms)
        in_domain  = self._looks_like_ipl_question(q)

        # Special handling: final/qualifier/eliminator with "why" or "how" → verify premise first
        final_terms = ["final", "qualifier", "eliminator", "playoff", "semi final"]
        is_final_question = any(term in q for term in final_terms)
        
        # Premise verification: if question implies a match result fact, verify first
        premise_patterns = [
            r"\b(why|how)\s+did\s+\w+\s+(lose|win|beat|defeat)\b",
            r"\b(why|how)\s+did\s+\w+\s+\w+\s+(lose|win|beat|defeat)\b",
            r"\b(why|how)\s+(did\s+)?[\w\s]*?(final|qualifier|eliminator|playoff)\b",
        ]
        needs_premise_check = any(re.search(p, q) for p in premise_patterns)

        # PRIORITY: If question requires premise verification, do ONLY that first
        # Don't add search_docs until we know the premise is true
        if needs_premise_check:
            if is_final_question:
                plan.append({
                    "tool":       "query_data",
                    "tool_input": question,
                    "reason":     "Verifying factual premise (match participation, results) before answering.",
                })
            else:
                plan.append({
                    "tool":       "query_data",
                    "tool_input": question,
                    "reason":     "Verifying factual premise embedded in the question before answering.",
                })
            # Return early - let the planner decide next steps based on premise result
            return plan

        # If we reach here, there's no premise to verify
        if needs_web:
            plan.append({
                "tool":       "web_search",
                "tool_input": self._short_web_query(question),
                "reason":     "Question asks for recent/current information.",
            })

        if needs_data and not needs_web and "query_data" not in {i["tool"] for i in plan}:
            plan.append({
                "tool":       "query_data",
                "tool_input": question,
                "reason":     "Question asks for structured IPL statistics or match results.",
            })

        if needs_docs and not needs_web and not needs_data:
            plan.append({
                "tool":       "search_docs",
                "tool_input": question,
                "reason":     "Question asks for narrative context from match reports.",
            })

        if not plan and in_domain:
            plan.append({
                "tool":       "search_docs",
                "tool_input": question,
                "reason":     "IPL-domain question — searching match reports.",
            })

        # Deduplicate while preserving order
        deduped: List[Dict[str, str]] = []
        seen: set = set()
        for item in plan:
            if item["tool"] not in seen:
                deduped.append(item)
                seen.add(item["tool"])
        return deduped

    # ── tool execution ────────────────────────────────────────────────────────

    def _call_tool(self, step_number: int, tool_name: str, tool_input: str, reason: str) -> Dict[str, Any]:
        started = time.time()
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
            output        = {"error": str(exc)}
            error_message = str(exc)

        return {
            "step":            step_number,
            "tool":            tool_name,
            "input":           tool_input,
            "reason":          reason,
            "output":          output,
            "error":           error_message,
            "latency_seconds": round(time.time() - started, 3),
        }

    # ── answer composition ────────────────────────────────────────────────────

    def _compose_final_answer(self, question: str, steps: List[Dict[str, Any]]) -> str:
        """
        Synthesise a fluent cited answer from tool results.
        - With OPENAI_API_KEY: calls the LLM to write a proper answer from the retrieved evidence.
        - Without key: builds a coherent structured summary from key facts in each tool result.

        The LLM is given ONLY the retrieved evidence — it must synthesize, not copy.
        """
        usable_steps = [s for s in steps if not s.get("error")]
        if not usable_steps:
            errors = "; ".join(s.get("error") or "unknown" for s in steps)
            return f"I could not answer from the available tools. Tool errors: {errors}"

        # ── Build a rich, structured evidence block for the LLM ──────────────
        evidence_parts: List[str] = []
        for step in usable_steps:
            tool   = step["tool"]
            output = step["output"]
            cites  = self._citations_from_output(tool, output)

            evidence_parts.append(f"--- Tool: {tool} | Input: {step['input']!r} ---")

            if tool == "query_data":
                result    = output.get("result")
                row_count = output.get("row_count", 0)
                if isinstance(result, list):
                    # Show up to 10 rows as a readable table
                    rows_text = json.dumps(result[:10], ensure_ascii=False, indent=2)
                    evidence_parts.append(f"Structured result ({row_count} rows):\n{rows_text}")
                elif result is not None:
                    evidence_parts.append(f"Structured result: {result}")
                else:
                    evidence_parts.append("No structured data returned.")

            elif tool == "search_docs":
                chunks = output.get("results", [])
                for i, chunk in enumerate(chunks[:3], 1):
                    score = chunk.get("score", 0)
                    cite  = chunk.get("citation", "")
                    text  = chunk.get("text", "")[:600]
                    evidence_parts.append(
                        f"Chunk {i} (score={score:.3f}, {cite}):\n{text}"
                    )

            elif tool == "web_search":
                results = output.get("results", [])
                for i, item in enumerate(results[:3], 1):
                    snippet = item.get("snippet", "")[:400]
                    url     = item.get("url", "")
                    evidence_parts.append(f"Web result {i} ({url}):\n{snippet}")

            if cites:
                evidence_parts.append("Source citations: " + " | ".join(cites[:5]))
            evidence_parts.append("")

        evidence = "\n".join(evidence_parts).strip()

        # ── LLM synthesis (primary path — single API call, no second round-trip) ────────────
        api_key = _get_api_key()
        if api_key:
            base_url = _get_base_url()
            synthesis_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert IPL cricket analyst. "
                        "The user asked a question and the agent retrieved evidence from tools. "
                        "Your job is to READ the evidence carefully and SYNTHESIZE a clear, "
                        "accurate answer — do NOT just paste the raw data.\n\n"
                        "Rules:\n"
                        "1. Answer in 2-5 sentences of fluent English prose.\n"
                        "2. Extract the key facts from the evidence and explain them in context.\n"
                        "3. Cite every factual claim inline using the source name, "
                        "e.g. (IPL_2023&2024_merged.pdf, p.47) or (iplmatches.csv).\n"
                        "4. If multiple tools provided data, integrate them into one coherent answer.\n"
                        "5. If the evidence contains a factual correction (e.g. team did not play "
                        "the final), state the correction clearly first.\n"
                        "6. Never guess or hallucinate — if the evidence is insufficient, say so.\n"
                        "7. Do not use bullet points. Write in paragraph form."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Retrieved Evidence:\n{evidence}\n\n"
                        "Now synthesize a clear, cited answer using only the evidence above."
                    ),
                },
            ]
            try:
                body = json.dumps({
                    "model":       self.config.model,
                    "messages":    synthesis_messages,
                    "temperature": 0.1,
                    "max_tokens":  500,
                }).encode("utf-8")
                req = request.Request(
                    base_url,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type":  "application/json",
                        "User-Agent":    "python-groq/0.9.0",
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=self.config.request_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                pass  # fall through to structured fallback

        # ── Structured fallback (no API key) ─────────────────────────────────
        # Extract key facts and build a coherent answer without an LLM
        facts: List[str] = []
        for step in usable_steps:
            tool   = step["tool"]
            output = step["output"]
            cites  = self._citations_from_output(tool, output)
            cite_s = f" [{' | '.join(cites[:2])}]" if cites else f" [{tool}]"

            if tool == "query_data":
                result = output.get("result")
                if isinstance(result, list) and result:
                    # Summarise the first few rows naturally
                    first = result[0]
                    pairs = ", ".join(
                        f"{k.replace('_', ' ')}: {v}"
                        for k, v in list(first.items())[:5]
                        if v is not None
                    )
                    row_count = output.get("row_count", len(result))
                    if row_count == 1:
                        facts.append(f"{pairs}{cite_s}")
                    else:
                        facts.append(
                            f"Top result — {pairs} (and {row_count - 1} more records){cite_s}"
                        )
                elif result is not None:
                    facts.append(f"{result}{cite_s}")

            elif tool == "search_docs":
                for item in output.get("results", [])[:2]:
                    raw       = item.get("text", "")
                    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
                    snippet   = " ".join(sentences[:3])[:350]
                    facts.append(f"{snippet}{cite_s}")
                    break  # only first chunk

            elif tool == "web_search":
                for item in output.get("results", [])[:1]:
                    snippet = item.get("snippet", "")[:350]
                    url     = item.get("url", tool)
                    facts.append(f"{snippet} [{url}]")

        if not facts:
            return "The tools returned data but I could not extract a clear answer. Set OPENAI_API_KEY for full synthesis."

        # Join facts into a paragraph
        answer = " ".join(facts)
        return f"{answer}\n\n[Tip: Set OPENAI_API_KEY in .env for a fully synthesized, fluent answer.]"

    # ── relevance check ───────────────────────────────────────────────────────

    def _relevance_summary(self, tool_name: Optional[str], output: Any) -> Dict[str, Any]:
        """
        Returns relevance metadata for each tool's output.
        Used by both heuristic and LLM planners to decide whether to fall back.
        """
        if not isinstance(output, dict) or output.get("error"):
            return {"relevant": False, "reason": "tool returned an error or no results"}

        if tool_name == "search_docs":
            results   = output.get("results", [])
            if not results:
                return {"relevant": False, "max_score": 0.0, "reason": "no chunks returned"}
            scores    = [r.get("score", 0.0) for r in results]
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            relevant  = max_score >= RELEVANCE_THRESHOLD
            return {
                "relevant":  relevant,
                "max_score": round(max_score, 4),
                "avg_score": round(avg_score, 4),
                "reason":    "scores above threshold" if relevant else (
                    f"max score {max_score:.3f} < {RELEVANCE_THRESHOLD} — chunks are likely off-topic"
                ),
            }

        if tool_name == "query_data":
            result    = output.get("result")
            row_count = output.get("row_count", 0)
            if result is None or result == 0 or row_count == 0:
                return {"relevant": False, "reason": "query returned no matching rows"}
            return {"relevant": True, "reason": "query returned data"}

        if tool_name == "web_search":
            results = output.get("results", [])
            return {
                "relevant": len(results) > 0,
                "reason":   "web results found" if results else "no web results",
            }

        return {"relevant": True, "reason": "unknown tool"}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _summarize_result(self, output: Any) -> str:
        if not isinstance(output, dict):
            return str(output)
        if output.get("error"):
            return f"Error: {output['error']}"
        if "results" in output:
            rows = []
            for item in output.get("results", [])[:3]:
                title   = item.get("title") or item.get("citation") or item.get("source") or "result"
                snippet = item.get("snippet") or item.get("text") or ""
                snippet = " ".join(str(snippet).split())[:300]
                rows.append(f"- {title}: {snippet}")
            return "\n".join(rows) if rows else "No results."
        result = output.get("result")
        if isinstance(result, list):
            return json.dumps(result[:5], ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False)

    def _citations_from_output(self, tool_name: Optional[str], output: Any) -> List[str]:
        if not isinstance(output, dict) or output.get("error"):
            return []
        if tool_name == "query_data":
            source   = output.get("source") or {}
            citation = source.get("citation")
            dataset  = source.get("dataset")
            rows     = source.get("rows_used")
            if citation:
                return [citation]
            if dataset:
                return [f"{dataset}, rows {rows}"]
            return []
        if tool_name in {"search_docs", "web_search"}:
            return [
                item["citation"]
                for item in output.get("results", [])
                if item.get("citation")
            ]
        return []

    def _collect_citations(self, steps: List[Dict[str, Any]]) -> List[str]:
        citations: List[str] = []
        for step in steps:
            citations.extend(self._citations_from_output(step.get("tool"), step.get("output")))
        seen: set = set()
        deduped   = []
        for c in citations:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        return deduped

    def _finish_refusal(self, trace: Dict[str, Any], reason: str, started_at: float) -> Dict[str, Any]:
        trace["refused"]          = True
        trace["final_answer"]     = reason
        trace["citations"]        = self._collect_citations(trace["steps"])
        trace["duration_seconds"] = round(time.time() - started_at, 3)
        return trace

    def _direct_answer(self, question: str) -> Optional[str]:
        if re.fullmatch(r"[0-9+\-*/().\s]+", question):
            try:
                tree = ast.parse(question, mode="eval")
                allowed = (
                    ast.Expression, ast.BinOp, ast.UnaryOp,
                    ast.Add, ast.Sub, ast.Mult, ast.Div,
                    ast.FloorDiv, ast.Mod, ast.Pow,
                    ast.USub, ast.UAdd, ast.Constant,
                )
                if all(isinstance(node, allowed) for node in ast.walk(tree)):
                    value = eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, {})
                    return f"{question} = {value}"
            except Exception:
                return None
        return None

    def _safety_refusal_reason(self, question: str) -> Optional[str]:
        q = question.lower()
        if re.search(r"\b(should i|recommend|advice)\b", q) and \
           re.search(r"\b(invest|buy|sell|stock|bet|gamble|fantasy)\b", q):
            return "I cannot provide investment, betting, or gambling advice."
        if re.search(r"\b(guaranteed|sure shot|fixed match)\b", q):
            return "I cannot help with gambling, match-fixing, or guaranteed-outcome claims."
        return None

    def _verify_premise(self, question: str) -> Optional[str]:
        """
        Verify factual premises in questions before answering.
        Returns correction message if premise is false, None if premise is valid.
        """
        q = question.lower()
        
        # Pattern: "why/how did X win/lose/won/lost [the] [year] final/match"
        # Examples: "Why did MI lose the 2023 final?", "How did GT win 2023 final?"
        match = re.search(
            r'\b(why|how|what|when)\s+(?:did|was|were)\s+(\w+)\s+(win|lose|won|lost|victory|defeat|performance)\s+(?:the\s+)?(\d{4})?\s*(final|match)?',
            q
        )
        
        if not match:
            return None
        
        question_type = match.group(1)  # why, how, what, when
        team_abbr = match.group(2).upper()  # MI, CSK, GT, etc.
        outcome = match.group(3)  # win, lose, won, lost
        year = match.group(4)  # 2023, 2024
        match_type = match.group(5)  # final, match
        
        # Only verify for final-related questions with a year
        if not (match_type == "final" and year):
            return None
        
        # Map team abbreviations to full names
        team_map = {
            "mi": "Mumbai Indians",
            "csk": "Chennai Super Kings",
            "rcb": "Royal Challengers Bangalore",
            "kkr": "Kolkata Knight Riders",
            "gt": "Gujarat Titans",
            "srh": "Sunrisers Hyderabad",
            "dc": "Delhi Capitals",
            "rr": "Rajasthan Royals",
            "pbks": "Punjab Kings",
            "lsg": "Lucknow Super Giants",
        }
        
        team_full = team_map.get(team_abbr.lower(), team_abbr)
        
        # Verify: Who actually won the final in this year?
        try:
            verify_result = query_data_module.query_data(f"Who won the IPL {year} final?")
            
            if isinstance(verify_result, dict) and not verify_result.get("error"):
                result_data = verify_result.get("result")
                
                # Extract winner and teams from result
                actual_winner = None
                teams_in_final = []
                
                if isinstance(result_data, str):
                    actual_winner = result_data
                elif isinstance(result_data, dict):
                    actual_winner = result_data.get("winner")
                elif isinstance(result_data, list) and len(result_data) > 0:
                    first_match = result_data[0]
                    if isinstance(first_match, dict):
                        actual_winner = first_match.get("winner")
                        team1 = first_match.get("team1")
                        team2 = first_match.get("team2")
                        if team1:
                            teams_in_final.append(team1)
                        if team2:
                            teams_in_final.append(team2)
                
                if actual_winner:
                    actual_winner_lower = actual_winner.lower()
                    team_full_lower = team_full.lower()
                    
                    # Check if the team in question actually played in the final
                    team_played = (team_full_lower in actual_winner_lower or 
                                 any(team_full_lower in t.lower() for t in teams_in_final))
                    
                    # Check if the outcome matches (win vs lose)
                    is_win_question = outcome in ["win", "won", "victory"]
                    is_lose_question = outcome in ["lose", "lost", "defeat"]
                    team_won = team_full_lower in actual_winner_lower
                    
                    # False premise cases:
                    # 1. Team didn't play in the final at all
                    if not team_played:
                        if teams_in_final:
                            teams_str = " vs ".join(teams_in_final)
                        else:
                            teams_str = f"{actual_winner} (winner) vs another team"
                        
                        return (
                            f"{team_full} did not play in the IPL {year} final. "
                            f"The {year} final was between {teams_str}, with {actual_winner} winning. "
                            f"Would you like to know about {team_full}'s performance in the {year} season?"
                        )
                    
                    # 2. Team played but the outcome is wrong (asking why they won when they lost, or vice versa)
                    if is_win_question and not team_won:
                        return (
                            f"{team_full} did not win the IPL {year} final. "
                            f"{actual_winner} won the {year} final. "
                            f"Would you like to know about {team_full}'s performance in the final or their {year} season?"
                        )
                    
                    if is_lose_question and team_won:
                        return (
                            f"{team_full} did not lose the IPL {year} final - they won it! "
                            f"{team_full} defeated their opponent to win the {year} IPL championship. "
                            f"Would you like to know about their victory?"
                        )
        
        except Exception:
            # If verification fails, don't block the question - let it proceed normally
            pass

        return None

    def _looks_like_ipl_question(self, q: str) -> bool:
        terms = [
            "ipl", "cricket", "kohli", "dhoni", "bumrah", "shami",
            "csk", "rcb", "mi", "kkr", "srh", "gt", "lsg",
            "punjab kings", "rajasthan royals", "delhi capitals",
            "chennai super kings", "mumbai indians",
        ]
        return any(t in q for t in terms)

    def _short_web_query(self, question: str) -> str:
        words    = re.findall(r"[A-Za-z0-9]+", question)
        stop     = {"what", "who", "is", "are", "the", "of", "in", "for", "to", "a", "an", "and"}
        filtered = [w for w in words if w.lower() not in stop]
        return " ".join((filtered or words)[:10])


# ── CLI entry point ───────────────────────────────────────────────────────────

def _format_step_result(step: Dict[str, Any]) -> str:
    """Produce a compact, human-readable summary of one tool's output for the trace."""
    tool   = step["tool"]
    output = step.get("output", {})

    if not isinstance(output, dict) or output.get("error"):
        err = output.get("error", "unknown error") if isinstance(output, dict) else str(output)
        return f"  ERROR: {err}"

    if tool == "query_data":
        result    = output.get("result")
        row_count = output.get("row_count", 0)
        source    = output.get("source") or {}
        dataset   = source.get("dataset", "")
        calc      = source.get("calculation", "")
        if isinstance(result, list):
            # Show first row as compact key=value
            rows_shown = result[:3]
            rows_text  = "; ".join(
                "{ " + ", ".join(f"{k}: {v}" for k, v in row.items() if k != "_source_row") + " }"
                for row in rows_shown
            )
            suffix = f" ... (+{row_count - len(rows_shown)} more)" if row_count > len(rows_shown) else ""
            return (
                f"  result=[{rows_text}{suffix}]\n"
                f"  source={dataset}  rows={row_count}  calc='{calc}'"
            )
        return (
            f"  result={result!r}\n"
            f"  source={dataset}  rows={row_count}  calc='{calc}'"
        )

    if tool == "search_docs":
        results = output.get("results", [])
        lines   = []
        for r in results[:3]:
            snippet = " ".join(r.get("text", "").split())[:200]
            lines.append(
                f"  chunk {r['rank']}: score={r.get('score', 0):.3f}  "
                f"source={r.get('citation', '')}\n"
                f"    text=\"{snippet}...\""
            )
        return "\n".join(lines) if lines else "  No chunks returned."

    if tool == "web_search":
        results = output.get("results", [])
        lines   = []
        for i, r in enumerate(results[:3], 1):
            snippet = r.get("snippet", "")[:200]
            url     = r.get("url", "")
            lines.append(f"  result {i}: url={url}\n    snippet=\"{snippet}\"")
        return "\n".join(lines) if lines else "  No web results."

    return f"  {json.dumps(output, ensure_ascii=False)[:300]}"


def run_agent(question: str, planner: str = "auto", as_json: bool = False) -> str:
    agent = AgenticRAGAgent(AgentConfig(planner=planner))
    trace = agent.run(question)

    if as_json:
        return json.dumps(trace, indent=2, ensure_ascii=False)

    # ── Required structured trace format ──────────────────────────────────────
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"Question: {trace['question']}")
    lines.append("=" * 64)

    if trace["steps"]:
        lines.append("")
        lines.append("Trace Log:")
        lines.append("-" * 40)
        for step in trace["steps"]:
            tool_label = step["tool"]
            inp        = step["input"]
            latency    = step.get("latency_seconds", 0)
            reason     = step.get("reason", "")

            lines.append(f"Step {step['step']}: tool={tool_label}  input={inp!r}")
            if reason:
                lines.append(f"  reason: {reason}")
            lines.append(_format_step_result(step))
            lines.append(f"  latency: {latency}s")
            lines.append("")
    else:
        lines.append("")
        lines.append("(No tool calls — answered directly)")
        lines.append("")

    lines.append("-" * 40)
    lines.append("Final Answer:")
    lines.append(trace["final_answer"] or "[No answer produced]")
    lines.append("")

    if trace["citations"]:
        lines.append("Citations:")
        for c in trace["citations"]:
            lines.append(f"  - {c}")
        lines.append("")

    cap = trace["max_steps"]
    used = trace["steps_used"]
    lines.append(f"Steps used: {used} / {cap} max")

    if trace.get("refused"):
        lines.append("[REFUSED — agent could not answer within constraints]")
    if trace.get("error"):
        lines.append(f"[ERROR: {trace['error']}]")

    lines.append("=" * 64)
    return "\n".join(lines)
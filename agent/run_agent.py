from __future__ import annotations

import argparse
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_core import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IPL Agentic RAG agent.")
    parser.add_argument("question", nargs="+", help="Question to ask the agent.")
    parser.add_argument(
        "--planner",
        choices=["auto", "llm", "heuristic"],
        default="auto",
        help="Planner mode. auto uses LLM when OPENAI_API_KEY is set, otherwise heuristic.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full trace as JSON.")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(run_agent(question, planner=args.planner, as_json=args.json))


if __name__ == "__main__":
    main()

# IPL Agentic RAG Agent

This folder contains the hand-written agent loop that connects the three assignment tools:

- `search_docs`: searches the IPL 2023/2024 PDF match-report corpus.
- `query_data`: queries the structured IPL CSV datasets with pandas.
- `web_search`: searches the live web through Tavily.

The loop is implemented in `agent_core.py`. It enforces the assignment's hard cap of 8 tool calls per question, records every tool call in a structured trace, and refuses unsafe or unanswerable questions instead of guessing.

## Run

From the project root:

```bash
python agent/run_agent.py "Who won Match 1 of IPL 2024?" --planner heuristic
```

Full JSON trace:

```bash
python agent/run_agent.py "How many runs did Virat Kohli score in 2023 and what did the report say about his batting?" --planner heuristic --json
```

To use the LLM planner, create a local `.env` file in the project root or `agent/` folder:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Then run:

```bash
python agent/run_agent.py "How many runs did Virat Kohli score in 2023 and what did the report say about his batting?" --planner llm --json
```

Use `--planner auto` to use the LLM when `OPENAI_API_KEY` is available and the heuristic fallback otherwise.

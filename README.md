# IPL Agentic RAG

A small reasoning agent that answers questions about IPL 2023 and IPL 2024 by deciding which tool to call, calling it, and composing a grounded answer with citations. It handles questions that require combining structured statistics with match-report narratives and live web results.

The agent loop is hand-written — a plain Python `while` loop with a hard cap of 8 tool calls. No black-box framework wraps it.

---

## What it does

Ask it a question. It routes to one or more of three tools:

| Tool | Source | Use when |
|------|--------|----------|
| `search_docs` | IPL 2023 & 2024 PDF match reports | Narratives, player performances, match context, playoff details |
| `query_data` | 3 IPL CSVs (145 matches, 34 966 ball-by-ball rows, 772 players) | Runs, wickets, strike rates, win counts, head-to-head, season results |
| `web_search` | Tavily live web search | Current squads, injuries, recent news, 2025 season updates |

It refuses questions it cannot answer — investment advice, betting, match-fixing, and anything outside the IPL domain — without calling any tool. It stops cleanly at 8 tool calls instead of guessing.

---

## Project structure

```
IPL-AGENTIC-RAG-CHATBOT/
├── agent/
│   ├── agent_core.py        ← Hand-written agent loop (the core of the project)
│   └── run_agent.py         ← CLI entry point
├── search_docs/
│   ├── search_docs_tool.py  ← Vector DB RAG (FAISS + sentence-transformers) over IPL PDF corpus
│   ├── search_docs_index.pkl← Cached FAISS vector index
│   └── search_docs_rag.ipynb← Tool test notebook
├── query_data/
│   ├── query_data_tool.py   ← Pandas queries over 3 IPL CSVs
│   └── query_data.ipynb     ← Tool test notebook
├── web_search/
│   ├── web_search_tool.py   ← Tavily web search wrapper
│   └── wed_search.ipynb     ← Tool test notebook
├── evaluate.py              ← Runs the 20-question evaluation set
├── DESIGN.md                ← Agent loop design, tool schemas, safety design
├── EVALUATION.md            ← 20-question eval set with actual agent outputs
├── .env.example             ← API key template
├── .gitignore
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/VishwaS-AM124/IPL-AGENTIC-RAG-CHATBOT.git
cd IPL-AGENTIC-RAG-CHATBOT
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=your_openai_api_key_here   # optional — only needed for LLM planner
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=your_tavily_api_key_here   # optional — only needed for web_search tool
```

**Both keys are optional.** The agent runs fully offline in heuristic mode using `--planner heuristic`. Only `web_search` queries require Tavily. Only `--planner llm` requires OpenAI.

**Data files required** (not committed — too large for GitHub):

Place these in their respective tool folders before running:

```
search_docs/IPL_2023&2024_merged.pdf   ← merged Wikipedia season PDFs
query_data/iplmatches.csv              ← match-level results (145 rows, 20 cols)
query_data/ipl2324.csv                 ← ball-by-ball data (34 966 rows, 65 cols)
query_data/players-data-updated.csv    ← player profiles (772 rows, 8 cols)
```

**First-run setup for search_docs:**

On the first run, the `search_docs` tool will:
1. Download the sentence-transformers model `all-MiniLM-L6-v2` (~90MB) from HuggingFace
2. Build the FAISS vector index from the PDF (~30-60 seconds)
3. Cache the index to `search_docs/search_docs_index.pkl` for fast loading

Subsequent runs load the cached index in 1-2 seconds.

---

## Run the agent

```bash
# Single-tool question (heuristic planner, no API key needed)
python agent/run_agent.py "Who won the IPL 2023 final?" --planner heuristic

# Multi-tool question (stats + narrative)
python agent/run_agent.py "How many runs did Virat Kohli score in 2023 and what did the match report say about his batting?" --planner heuristic

# Live web question (requires Tavily key)
python agent/run_agent.py "What is the current IPL 2025 points table?" --planner heuristic

# Full JSON trace output
python agent/run_agent.py "Who won Match 1 of IPL 2024?" --planner heuristic --json

# LLM planner (requires OpenAI key)
python agent/run_agent.py "Compare CSK and MI head-to-head and explain what the report said about their rivalry" --planner llm --json
```

---

## Run the evaluation set

```bash
python evaluate.py
```

Runs all 20 questions from `EVALUATION.md` and prints pass/fail per category. Results are also written to `eval_output.json`.

---

## Test each tool independently

Each tool can be called from the command line before running the agent — as the assignment requires:

```bash
# Test search_docs (downloads embedding model on first run, builds FAISS index)
python search_docs/search_docs_tool.py

# Test query_data
python query_data/query_data_tool.py

# Test web_search (requires TAVILY_API_KEY in .env)
python web_search/web_search_tool.py
```

Or open the corresponding notebook for interactive testing.

---

## Example traces

**Single-tool — structured stats**

```
Question: How many runs did Virat Kohli score in 2023?

Step 1: tool=query_data  input='How many runs did Virat Kohli score in 2023?'
         result=639  source=ipl2324.csv; 17 matching rows  latency=0.04s

Final Answer: Virat Kohli scored 639 runs in IPL 2023.
Citations: ipl2324.csv; 17 matching data row(s)
Steps used: 1 / 8 max
```

**Multi-tool — stats + narrative**

```
Question: How many runs did Kohli score in 2023 and what did the report say about his batting?

Step 1: tool=query_data  input='runs of Kohli 2023'
         result=639  source=ipl2324.csv; 17 rows  latency=0.04s

Step 2: tool=search_docs  input='Virat Kohli batting 2023'
         result=IPL_2023&2024_merged.pdf, p.12  score=0.8542  latency=0.02s

Final Answer: Kohli scored 639 runs in IPL 2023 (query_data). The match report
             describes his batting as consistently dominant through the tournament...
Citations: ipl2324.csv; 17 row(s) | IPL_2023&2024_merged.pdf, p.12
Steps used: 2 / 8 max
```

**Graceful refusal — unsafe question**

```
Question: Which team should I bet on in IPL 2025?

Final Answer: I cannot provide investment, betting, or gambling advice.
Citations: (none)
Steps used: 0 / 8 max
```

**Hard cap firing**

```
Question: What is the exact ball-by-ball trajectory of every over in every match of IPL 2023?

... [8 tool calls made] ...

Final Answer: I could not answer confidently within the 8 tool-call limit,
             so I am stopping instead of guessing.
Steps used: 8 / 8 max
```

---

## Design decisions

Full detail in `DESIGN.md`. Short version:

- **Vector DB RAG with FAISS** — Uses sentence-transformers (`all-MiniLM-L6-v2`) for semantic embeddings and FAISS IndexFlatIP for cosine similarity search. Index cached to disk with versioning. First run builds index (30-60s), subsequent runs load from cache (1-2s).
- **Dual planner** — LLM planner when OpenAI key is set, keyword heuristic otherwise. Same loop, different decision function.
- **Hard cap in code** — `while trace["steps_used"] < 8`. Not configurable above 8. Fires correctly on loop-inducing questions.
- **Citations from output, not from LLM** — citation strings are extracted from each tool's output dict. The LLM never generates citation text, so hallucinated citations are not possible.
- **Tool descriptions written for the LLM** — each tool's `TOOL_DESCRIPTION` constant says when to use it and, critically, when not to.

---

## Known failure modes

These are real failures observed during evaluation — not hypothetical:

**1. Heuristic misroutes stats questions that use narrative keywords**
Questions like "What was Kohli's batting performance in 2023?" contain "batting" (a doc_terms word) and get routed to `search_docs` first, when `query_data` would answer it faster. The LLM planner handles this correctly; the heuristic does not.
*Specific fix:* Add a `stats_override` check — if the question also matches a player-name + number pattern, prefer `query_data` even when doc_terms are present.

**2. `query_data` fails on unseen player name variants**
"How many runs did Virat score?" without the surname fails player matching. The alias resolver covers `player_name`, `player_full_name`, and `player_name2` but not arbitrary nicknames.
*Specific fix:* Add a fuzzy first-name match as a fallback in `_find_player_in_query`.

**3. `search_docs` uses semantic vector search for better relevance**
The tool now uses sentence-transformers for semantic embeddings and FAISS for vector similarity search. This provides better semantic understanding compared to keyword-based retrieval. The system includes metadata re-ranking to boost relevant chunks based on season, match number, teams, players, and context (batting/bowling). First run downloads the embedding model (~90MB) and builds the index (30-60s), but subsequent queries are fast (20-50ms).

**4. Hardcoded paths in test notebooks**
The three test notebooks (`query_data.ipynb`, `search_docs_rag.ipynb`, `wed_search.ipynb`) use absolute Windows paths for loading tool modules. These must be changed to relative paths before running on another machine. The agent itself (`agent_core.py`) uses relative paths correctly.

---

## AI tool disclosure

AI coding assistants (Claude, GitHub Copilot) were used for code suggestions, debugging, and documentation. All design decisions — the loop structure, BM25 retrieval, dual-planner design, citation system, and safety layers — were made by the candidate and can be explained line by line. No high-level agent framework (`initialize_agent`, LangChain, LangGraph) was used for the agent loop.

---

## Corpus details

| Source | Type | Size | Covers |
|--------|------|------|--------|
| `IPL_2023&2024_merged.pdf` | Unstructured | ~200 pages | Wikipedia match-by-match reports for all 74 IPL 2023 matches and 74 IPL 2024 matches |
| `iplmatches.csv` | Structured | 145 rows × 20 cols | Match-level results: teams, venue, winner, toss, player of match |
| `ipl2324.csv` | Structured | 34 966 rows × 65 cols | Ball-by-ball data: batter, bowler, runs, wickets, extras, over, innings |
| `players-data-updated.csv` | Structured | 772 rows × 8 cols | Player profiles: batting style, bowling style, fielding position |
| Tavily web search | Live | Top 3 results per query | Current standings, squad updates, injury news, 2025 season |

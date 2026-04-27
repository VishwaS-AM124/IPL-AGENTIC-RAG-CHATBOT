# IPL Agentic RAG Chatbot - System Design

## Overview

An intelligent question-answering system for IPL (Indian Premier League) cricket data that combines three specialized retrieval tools through an autonomous agent architecture. The system handles structured data queries, unstructured document search, and live web information retrieval.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Question                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Core (agent_core.py)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Decision Loop (max 8 tool calls)                         │  │
│  │  • Planner: LLM (GPT-4o-mini) or Heuristic               │  │
│  │  • Tool Selection & Routing                               │  │
│  │  • Answer Composition                                     │  │
│  │  • Trace Recording                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────┬──────────────┬──────────────┬─────────────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Tool 1  │   │  Tool 2  │   │  Tool 3  │
    └──────────┘   └──────────┘   └──────────┘
```

---

## Component Breakdown

### 1. Agent Core (`agent/agent_core.py`)

**Purpose**: Orchestrates tool selection, execution, and answer synthesis.

**Key Components**:

#### 1.1 AgenticRAGAgent Class
- **Configuration**: Model selection, max tool calls (hard limit: 8), temperature
- **Main Loop**: `run(question)` → returns structured trace with answer
- **Safety**: Refuses unsafe queries (investment advice, gambling, match-fixing)

#### 1.2 Planning Strategies

**A. LLM Planner** (when `OPENAI_API_KEY` available)
- Uses GPT-4o-mini for intelligent tool routing
- JSON-structured decision making
- System prompt enforces strict tool usage rules:
  - `query_data` for ALL numeric/statistical queries
  - `search_docs` ONLY for explanations/narratives
  - `web_search` ONLY for current/live information
- Compact step summaries to manage context

**B. Heuristic Planner** (fallback)
- Rule-based keyword matching
- Pattern detection for query classification:
  - Web terms: "current", "latest", "today", "injury", "squad"
  - Data terms: "how many", "count", "top", "runs", "wickets", "winner"
  - Doc terms: "why", "explain", "describe", "tactical", "performance analysis"
- Builds execution plan upfront, executes sequentially

#### 1.3 Tool Execution
- Isolated try-catch per tool call
- Latency tracking
- Error capture and propagation
- Step recording with input/output/reason

#### 1.4 Answer Composition

**LLM Composition** (preferred):
- Sends tool results to GPT-4o-mini
- Natural language synthesis
- Automatic source attribution

**Heuristic Composition** (fallback):
- Template-based formatting
- Result type detection (scalar, list, dict, DataFrame)
- Manual source extraction

#### 1.5 Trace Structure
```python
{
    "question": str,
    "planner": "llm" | "heuristic",
    "max_steps": 8,
    "steps": [
        {
            "step": int,
            "tool": str,
            "input": str,
            "reason": str,
            "output": dict,
            "error": str | None,
            "latency_seconds": float
        }
    ],
    "final_answer": str,
    "citations": [str],
    "steps_used": int,
    "refused": bool,
    "error": str | None,
    "duration_seconds": float
}
```

---

### 2. Tool 1: query_data (`query_data/query_data_tool.py`)

**Purpose**: Query structured IPL CSV datasets using pandas.

**Data Sources**:
- `iplmatches.csv`: Match-level data (teams, winners, dates, venues)
- `ipl2324.csv`: Ball-by-ball data (batters, bowlers, runs, wickets)
- `players-data-updated.csv`: Player profiles (names, styles, positions)

**Query Capabilities**:

| Query Type | Example | Implementation |
|------------|---------|----------------|
| Match wins | "How many matches did CSK win in 2023?" | Filter by winner + season |
| Player runs | "Total runs by Virat Kohli in 2024" | Group by batter, sum runs_batter |
| Strike rate | "Kohli's strike rate in 2023" | (runs_batter / valid_ball) × 100 |
| Wickets | "Bumrah's wickets in 2024" | Filter bowler_wicket=1, count |
| Top scorers | "Top 5 run scorers in 2023" | Group by batter, sort descending |
| Top bowlers | "Top 10 wicket takers" | Group by bowler, count wickets |
| Head-to-head | "MI vs CSK record" | Filter team1/team2 participation |
| Final winner | "Who won IPL 2024 final?" | Filter match_type=Final |
| Player info | "Virat Kohli batting style" | Lookup in players_df |

**Key Features**:
- **Safe execution**: No eval(), no SQL injection risk
- **Team alias resolution**: "csk" → "Chennai Super Kings"
- **Player name matching**: Fuzzy matching with aliases (full name, last name)
- **Season filtering**: Automatic extraction from query (e.g., "2023", "2024")
- **Source tracking**: Records CSV file, row numbers, filters used
- **Citation generation**: Structured metadata for attribution

**Output Format**:
```python
{
    "result": scalar | list[dict] | dict,  # Query result
    "columns": [str],                       # Column names (if tabular)
    "row_count": int,                       # Number of rows
    "source": {
        "dataset": str,                     # CSV filename
        "filters": dict,                    # Applied filters
        "row_count_used": int,              # Rows matched
        "rows_used": [int],                 # CSV line numbers
        "citation": str,                    # Human-readable citation
        "calculation": str                  # Aggregation performed
    },
    "error": str | None
}
```

---

### 3. Tool 2: search_docs (`search_docs/search_docs_tool.py`)

**Purpose**: Semantic search over IPL 2023/2024 match report PDF using vector RAG.

**Data Source**:
- `IPL_2023&2024_merged.pdf`: Unstructured match narratives, player performances, tactical analysis

**Architecture**:

#### 3.1 Document Processing Pipeline
```
PDF → Pages → Match Chunks + Paragraph Chunks → Embeddings → FAISS Index
```

**Chunking Strategy**:
1. **Match-level chunks**: Full match reports (regex-based extraction)
   - Pattern: `MATCH \d+` markers
   - Metadata: season, match_num, teams, is_final, is_playoff, scorecard
2. **Paragraph-level chunks**: Sliding window (3 sentences, step=2)
   - Minimum 180 characters
   - Inherits parent match metadata

**Metadata Extraction**:
- Season: 2023 or 2024
- Match number: Extracted from "MATCH 74" labels
- Teams: Pattern matching against known team names
- Match type: Final, Qualifier, Eliminator detection
- Page number: Character offset → page mapping

#### 3.2 Vector Embedding System

**Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- Dimensions: 384
- Speed: ~20ms per query
- Quality: Good semantic understanding for cricket domain

**Embedding Process**:
1. Concatenate chunk text + metadata (season, match, teams, scorecard)
2. Encode with sentence-transformers
3. L2-normalize for cosine similarity via inner product

**Vector Index**: FAISS `IndexFlatIP`
- Inner product on normalized vectors = cosine similarity
- Exact search (no approximation)
- Serialized to `search_docs_index.pkl` for fast loading

#### 3.3 Retrieval Pipeline

**Step 1: Fast-path metadata lookup**
- Direct match for season + match_num queries
- Instant return for "IPL 2024 final" type queries
- Score: 100.0 (metadata match)

**Step 2: Vector similarity search**
- Embed query with same model
- FAISS search for top-K candidates (K = max(top_k × 10, 30))
- Returns cosine similarity scores

**Step 3: Metadata re-ranking**
- Boost scores based on:
  - Match number match: +4.0
  - Final match: +4.0
  - Playoff match: +1.5
  - Team mention: +1.0 per team
  - Player name match: +12.0 (exact), +5.0 per term mention
  - Bowling context: +8.0 (if query wants bowling)
  - Batting context: +6.0 (if query wants batting)
- Final score = cosine_similarity + (metadata_boost × 0.05)

**Step 4: Season filtering**
- Remove chunks from wrong season if specified

#### 3.4 Query Parsing
- Season extraction: "2023", "2024"
- Match number: "match 1", "first match", "opener"
- Team extraction: Aliases (csk, mi) + full names
- Player extraction: Capitalized multi-word names
- Intent detection: is_final, is_playoff, wants_bowling, wants_batting

#### 3.5 Caching Strategy
- Cache key: `(CACHE_VERSION, pdf_mtime, EMBEDDING_MODEL)`
- Invalidation: PDF change, code version bump, model change
- First run: 30-60s (build index)
- Subsequent runs: 1-2s (load from pickle)

**Output Format**:
```python
{
    "tool": "search_docs",
    "query": str,
    "results": [
        {
            "rank": int,
            "text": str,                    # Chunk text (max 1800 chars)
            "source": str,                  # PDF filename
            "page": int,                    # Page number
            "score": float,                 # Similarity + boost
            "season": int,                  # 2023 or 2024
            "match_num": int,               # Match number
            "chunk_type": "match" | "paragraph",
            "citation": str,                # "filename, p.X"
            "reason": str,                  # Retrieval method
            "truncated": bool               # Text was truncated
        }
    ],
    "result_count": int,
    "error": str | None
}
```

---

### 4. Tool 3: web_search (`web_search/web_search_tool.py`)

**Purpose**: Fetch current/live IPL information from the web via Tavily API.

**Use Cases**:
- Current squad information
- Recent injuries/transfers
- Live standings
- Latest news
- Rule changes
- Auction results

**API Integration**:
- Endpoint: `https://api.tavily.com/search`
- Authentication: `TAVILY_API_KEY` (from .env)
- Search depth: "advanced"
- Max results: 3 (configurable)

**Query Constraints**:
- Maximum 10 words (enforced)
- Timeout: 15 seconds (configurable)

**Result Processing**:
- Title cleaning (max 180 chars)
- Snippet extraction (max 500 chars)
- URL validation
- Publication date extraction (multiple field fallbacks)

**Output Format**:
```python
{
    "tool": "web_search",
    "query": str,
    "results": [
        {
            "title": str,
            "snippet": str,
            "url": str,
            "published_date": str | None,
            "citation": str              # "URL (date)"
        }
    ],
    "result_count": int,
    "error": str | None
}
```

**Error Handling**:
- Missing API key
- HTTP errors (with response detail)
- Timeout errors
- Invalid JSON responses
- Empty results

---

## Data Flow

### Example Query: "Who won the IPL 2024 final and what was the match report?"

```
1. User Input
   └─> "Who won the IPL 2024 final and what was the match report?"

2. Agent Core (Planner Decision)
   ├─> LLM Planner analyzes query
   ├─> Detects: needs structured data (winner) + narrative (report)
   └─> Plan: [query_data, search_docs]

3. Step 1: query_data
   ├─> Input: "Who won the IPL 2024 final"
   ├─> Filters: season=2024, match_type=Final
   ├─> Result: {"winner": "Kolkata Knight Riders", ...}
   └─> Latency: ~50ms

4. Step 2: search_docs
   ├─> Input: "IPL 2024 final match report"
   ├─> Fast-path: Direct metadata match (season=2024, is_final=True)
   ├─> Result: Match 74 chunk with full narrative
   └─> Latency: ~20ms (cached index)

5. Answer Composition (LLM)
   ├─> Combines: KKR winner + match narrative
   ├─> Synthesizes: Natural language answer
   └─> Adds citations: [iplmatches.csv, IPL_2023&2024_merged.pdf p.293]

6. Output
   └─> "Kolkata Knight Riders won the IPL 2024 final, defeating 
        Sunrisers Hyderabad by 8 wickets. KKR chased down SRH's 
        total of 113 with ease, scoring 114/2. Venkatesh Iyer 
        scored 52* and Andre Russell took 3-19.
        
        Sources: iplmatches.csv; IPL_2023&2024_merged.pdf, p.293"
```

---

## Configuration & Environment

### Required Environment Variables

**For LLM Planner** (optional, falls back to heuristic):
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # default
```

**For Web Search** (required for web_search tool):
```bash
TAVILY_API_KEY=tvly-...
```

### Configuration Files

**`.env` locations** (checked in order):
1. Project root: `/.env`
2. Agent directory: `/agent/.env`
3. Web search directory: `/web_search/.env`

---

## Tool Selection Logic

### LLM Planner Rules (System Prompt)
```
1. query_data MUST be used for ALL numeric/statistical queries
   - Includes: runs, wickets, tally, totals, counts, player stats, averages
   - ALWAYS use for: "What was [player] [stat] in [season/year]?"

2. search_docs MUST ONLY be used for explanations
   - NEVER use for numeric/statistical queries

3. web_search ONLY for current/live info
   - Use when query includes: "current", "latest", "today"

4. query_data has HIGH PRIORITY over search_docs

5. Choosing search_docs for a statistical query is WRONG
```

### Heuristic Planner Rules
```python
# Web search triggers
web_terms = ["rules", "current", "latest", "today", "recent", 
             "news", "live", "injury", "standings", "auction", "squad"]

# Structured data triggers
data_terms = ["how many", "count", "number of", "top", "most", 
              "highest", "total", "runs", "wickets", "tally", 
              "statistics", "stats", "strike rate", "average", 
              "winner", "won", "win", "matches", "season"]

# Document search triggers
doc_terms = ["why", "reason", "explain", "described", "report", 
             "describe", "summary", "tactical change", "context", 
             "performance analysis", "impactful player", "thriller"]

# Priority: web > data > docs (if multiple match)
```

---

## Safety & Constraints

### Hard Limits
- **Max tool calls**: 8 per question (assignment requirement)
- **Query length**: 10 words max for web_search
- **Result limits**: 3 results per tool (configurable)
- **Text truncation**: 1800 chars per search_docs result

### Safety Refusals
Agent refuses queries containing:
- Investment/betting advice: "should i invest", "bet", "gamble", "fantasy"
- Match-fixing: "guaranteed", "sure shot", "fixed match"

### Error Handling
- Tool failures don't crash agent (captured in trace)
- Empty results trigger fallback or refusal
- Timeout protection on all network calls
- Graceful degradation (LLM → heuristic)

---

## Performance Characteristics

### Latency Breakdown

| Component | First Run | Cached |
|-----------|-----------|--------|
| query_data | 50-100ms | 50-100ms |
| search_docs (index build) | 30-60s | - |
| search_docs (query) | - | 20-50ms |
| web_search | 1-3s | 1-3s |
| LLM planner decision | 500-1000ms | 500-1000ms |
| LLM answer composition | 1-2s | 1-2s |

### Typical Query Times
- Simple data query: 1-2s (heuristic) or 2-3s (LLM)
- Document search: 2-3s (after first run)
- Multi-tool query: 3-5s
- Web search query: 3-4s

### Resource Usage
- **Memory**: ~500MB (with loaded models)
- **Disk**: ~100MB (embeddings cache + models)
- **Network**: Only for LLM calls and web_search

---

## Dependencies

### Core
```
pandas          # Data manipulation
pypdf           # PDF text extraction
```

### Vector RAG
```
sentence-transformers  # Semantic embeddings
faiss-cpu             # Vector similarity search
numpy                 # Array operations
```

### Optional (for LLM features)
```
openai-api (via urllib)  # LLM planner & composition
```

---

## File Structure

```
IPL_CHATBOT/
├── agent/
│   ├── agent_core.py          # Main agent orchestration
│   ├── run_agent.py           # CLI entry point
│   └── README.md              # Agent documentation
│
├── query_data/
│   ├── query_data_tool.py     # Structured data queries
│   ├── iplmatches.csv         # Match-level data
│   ├── ipl2324.csv            # Ball-by-ball data
│   └── players-data-updated.csv  # Player profiles
│
├── search_docs/
│   ├── search_docs_tool.py    # Vector RAG implementation
│   ├── IPL_2023&2024_merged.pdf  # Match reports
│   └── search_docs_index.pkl  # Cached FAISS index
│
├── web_search/
│   ├── web_search_tool.py     # Tavily API integration
│   └── .env                   # TAVILY_API_KEY
│
├── .env                       # OPENAI_API_KEY (optional)
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
├── EVALUATION.md              # Evaluation criteria
└── DESIGN.md                  # This file
```

---

## Usage Examples

### CLI Usage
```bash
# Basic query (auto planner)
python agent/run_agent.py "Who won IPL 2024?"

# Force heuristic planner
python agent/run_agent.py "Top 5 run scorers in 2023" --planner heuristic

# Force LLM planner
python agent/run_agent.py "Explain Kohli's performance" --planner llm

# Get full JSON trace
python agent/run_agent.py "CSK vs MI head to head" --json
```

### Programmatic Usage
```python
from agent.agent_core import AgenticRAGAgent, AgentConfig

# Create agent
agent = AgenticRAGAgent(AgentConfig(
    planner="auto",
    model="gpt-4o-mini",
    max_tool_calls=8,
    temperature=0.0
))

# Run query
trace = agent.run("Who won the IPL 2024 final?")

# Access results
print(trace["final_answer"])
print(trace["citations"])
print(f"Used {trace['steps_used']} tool calls")
```

### Direct Tool Usage
```python
# Query structured data
from query_data.query_data_tool import query_data
result = query_data("How many matches did CSK win in 2023?")

# Search documents
from search_docs.search_docs_tool import search_docs
result = search_docs("IPL 2024 final match report")

# Web search
from web_search.web_search_tool import web_search
result = web_search("IPL 2024 current standings")
```

---

## Design Principles

1. **Tool Specialization**: Each tool has a clear, non-overlapping domain
2. **Safety First**: Hard limits, refusals, error isolation
3. **Traceability**: Every decision and tool call is recorded
4. **Graceful Degradation**: LLM → heuristic, cached → rebuild
5. **Citation Discipline**: Every answer includes source attribution
6. **Performance**: Caching, efficient indexing, timeout protection
7. **Maintainability**: Modular tools, clear interfaces, type hints

---

## Future Enhancements

### Potential Improvements
1. **Approximate vector search**: FAISS IVF for larger corpora
2. **Cross-encoder re-ranking**: Better top-K selection
3. **Query expansion**: LLM-based query reformulation
4. **Multi-hop reasoning**: Chain tool calls intelligently
5. **Streaming responses**: Real-time answer generation
6. **Tool result caching**: Avoid redundant calls
7. **Confidence scoring**: Uncertainty quantification
8. **User feedback loop**: Learn from corrections

### Scalability Considerations
- Current design handles ~1000 PDF pages efficiently
- For 10K+ pages: Switch to approximate FAISS index (IVF)
- For 100K+ pages: Consider distributed vector DB (Pinecone, Weaviate)
- For high QPS: Add Redis caching layer

---

## Conclusion

The IPL Agentic RAG Chatbot demonstrates a production-ready architecture for multi-modal question answering, combining:
- **Structured data retrieval** (pandas)
- **Semantic document search** (vector RAG)
- **Live web information** (API integration)
- **Intelligent orchestration** (LLM + heuristic planning)

The system prioritizes correctness, traceability, and safety while maintaining good performance through caching and efficient indexing.

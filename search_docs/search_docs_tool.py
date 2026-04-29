"""
search_docs_tool.py  —  Pure-vector document retrieval for IPL Agentic RAG.

Retrieval stack:
  - Chunking  : page-level + sliding-window paragraph chunks from the merged PDF
  - Embeddings: sentence-transformers all-MiniLM-L6-v2  (local, no API key)
  - Index     : FAISS IndexFlatIP  (cosine similarity via normalised vectors)
  - Ranking   : pure cosine similarity only — NO metadata boost, NO BM25, NO TF-IDF

Index is built once and cached to disk as two files:
  search_docs/faiss_index.bin   — FAISS binary index
  search_docs/faiss_meta.pkl    — chunk metadata list (parallel to index rows)

Delete both files to force a rebuild (e.g. after the PDF changes).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent
PDF_NAME   = "IPL_2023&2024_merged.pdf"
PDF_PATH   = DATA_DIR / PDF_NAME
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH  = DATA_DIR / "faiss_meta.pkl"

# ── retrieval settings ────────────────────────────────────────────────────────
EMBED_MODEL      = "all-MiniLM-L6-v2"   # 80 MB, runs locally, no API key
CHUNK_SENTENCES  = 5                     # sentences per sliding-window chunk
CHUNK_OVERLAP    = 2                     # sentence overlap between chunks
MIN_CHUNK_CHARS  = 120                   # discard tiny chunks
MAX_RESULTS      = 3
MAX_CHARS_OUTPUT = 800                   # max chars shown per result chunk

# ── tool description (read by the LLM for routing decisions) ──────────────────
TOOL_DESCRIPTION = (
    "Search the local IPL 2023 and IPL 2024 match-report PDF corpus using pure semantic "
    "vector search (FAISS cosine similarity). Use this tool for explanations, match context, "
    "player-performance narratives, tactical analysis, playoff or final report details, and "
    "any facts written in the unstructured PDF. Do not use it for current/live news or "
    "for structured aggregate statistics (runs totals, wicket counts, win records) that "
    "query_data can answer directly. Input is one natural-language question or phrase. "
    "Output is the top 3 semantically relevant text chunks with source filename, page "
    "number, cosine score, and citation."
)

# ── team aliases (used ONLY for hard season filter in parse_query) ────────────
TEAM_ALIASES: Dict[str, List[str]] = {
    "csk":  ["Chennai Super Kings"],
    "mi":   ["Mumbai Indians"],
    "rcb":  ["Royal Challengers Bangalore", "Royal Challengers Bengaluru"],
    "kkr":  ["Kolkata Knight Riders"],
    "rr":   ["Rajasthan Royals"],
    "dc":   ["Delhi Capitals"],
    "pbks": ["Punjab Kings"],
    "srh":  ["Sunrisers Hyderabad"],
    "gt":   ["Gujarat Titans"],
    "lsg":  ["Lucknow Super Giants"],
}

FULL_TEAM_NAMES = [name.lower() for names in TEAM_ALIASES.values() for name in names]

GENERIC_WORDS = {
    "Which", "Who", "What", "How", "When", "Where", "Match", "Game",
    "IPL", "Super", "Kings", "Royal", "Challengers", "Knight", "Riders",
    "Capitals", "Titans", "Giants", "Sunrisers", "League",
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ── text utilities ────────────────────────────────────────────────────────────

def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF loading ───────────────────────────────────────────────────────────────

def _load_pages() -> List[Dict[str, Any]]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place IPL_2023&2024_merged.pdf in the search_docs/ folder."
        )
    reader = PdfReader(str(PDF_PATH))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append({"page": idx, "text": text})
    return pages


# ── metadata extraction ───────────────────────────────────────────────────────

def _extract_season(text: str) -> Optional[int]:
    m = re.search(r"\b(2023|2024)\b", text)
    return int(m.group(1)) if m else None


def _extract_match_num(text: str) -> Optional[int]:
    m = re.search(r"\b(?:MATCH|Match)\s+(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_teams(text: str) -> List[str]:
    text_low = text.lower()
    found = []
    for full in FULL_TEAM_NAMES:
        if full in text_low:
            found.append(full.title())
    return list(dict.fromkeys(found))


def _is_final(text: str) -> bool:
    return bool(re.search(r"\bFINAL\b", text))


def _is_playoff(text: str) -> bool:
    return bool(re.search(r"\b(QUALIFIER|ELIMINATOR)\b", text, re.I))


# ── chunking ──────────────────────────────────────────────────────────────────

def _make_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Two chunk types per page:
      1. Full-page chunk  — broad context, good for topic questions
      2. Sliding-window paragraph chunks — specific passages for detail questions
    """
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_num  = page["page"]
        page_text = page["text"]
        season    = _extract_season(page_text)
        match_num = _extract_match_num(page_text)
        teams     = _extract_teams(page_text[:600])
        is_fin    = _is_final(page_text[:300])
        is_play   = _is_playoff(page_text[:300])

        base_meta = {
            "source":     PDF_NAME,
            "page":       page_num,
            "season":     season,
            "match_num":  match_num,
            "teams":      teams,
            "is_final":   is_fin,
            "is_playoff": is_play,
            "citation":   f"{PDF_NAME}, p.{page_num}",
        }

        # 1. Full-page chunk
        if len(page_text) >= MIN_CHUNK_CHARS:
            chunks.append({**base_meta, "text": page_text, "chunk_type": "page"})

        # 2. Sliding-window paragraph chunks
        sentences = [s.strip() for s in _SENT_RE.split(page_text) if s.strip()]
        step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
        for i in range(0, len(sentences), step):
            window = sentences[i : i + CHUNK_SENTENCES]
            para   = " ".join(window)
            if len(para) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                **base_meta,
                "text":       para,
                "chunk_type": "paragraph",
                "sent_start": i,
            })

    return chunks


# ── embedding + FAISS ─────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBED_MODEL}' (first run only)...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _embed(texts: List[str]) -> np.ndarray:
    vecs = _get_model().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")


def _build_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    print("Building FAISS vector index from PDF...")
    pages  = _load_pages()
    chunks = _make_chunks(pages)
    print(f"  {len(pages)} pages  ->  {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (~30 seconds on first run)...")
    vecs  = _embed(texts)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine similarity on normalised vectors
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved: {INDEX_PATH} and {META_PATH}")
    return index, chunks


def _load_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def _get_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """Load from disk or build if missing."""
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            return _load_index()
        except Exception:
            pass
    return _build_index()


# ── query parsing — used ONLY for hard season filter ─────────────────────────

def parse_query(query: str) -> Dict[str, Any]:
    """
    Extract structured metadata from the query.
    Used ONLY to drive the hard season-correctness filter during retrieval.
    NOT used for score manipulation.
    """
    q = _norm(query)

    season_m  = re.search(r"\b(2023|2024)\b", q)
    season    = int(season_m.group(1)) if season_m else None

    match_num = None
    m = re.search(r"\b(?:match|game|fixture)\s+(\d+)\b", q)
    if m:
        match_num = int(m.group(1))

    teams: List[str] = []
    for alias, full_names in TEAM_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            teams.extend(full_names)
    for name in FULL_TEAM_NAMES:
        if name in q:
            teams.append(name.title())
    teams = list(dict.fromkeys(teams))

    player_candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    players = [p for p in player_candidates
               if not any(part in GENERIC_WORDS for part in p.split())]

    return {
        "season":    season,
        "match_num": match_num,
        "teams":     teams,
        "players":   players,
    }


# ── domain guard ──────────────────────────────────────────────────────────────

_DOMAIN_TERMS = {
    "ipl", "cricket", "match", "final", "qualifier", "eliminator",
    "runs", "wickets", "batting", "bowling", "over", "innings",
    "csk", "mi", "rcb", "kkr", "rr", "dc", "pbks", "srh", "gt", "lsg",
    "season", "playoff",
}


def _is_in_domain(query: str, parsed: Dict[str, Any]) -> bool:
    if parsed["season"] or parsed["match_num"] or parsed["teams"] or parsed["players"]:
        return True
    q = _norm(query)
    for term in _DOMAIN_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", q):
            return True
    return any(name in q for name in FULL_TEAM_NAMES)


# ── deduplication ─────────────────────────────────────────────────────────────

def _dedup_key(chunk: Dict[str, Any]) -> str:
    return f"p{chunk['page']}:{chunk['text'][:80]}"


# ── public API ────────────────────────────────────────────────────────────────

def search_docs(query: str, top_k: int = MAX_RESULTS) -> Dict[str, Any]:
    """
    Pure vector search over the IPL 2023 & 2024 PDF corpus.

    Ranking is by FAISS cosine similarity ONLY — no BM25, no TF-IDF,
    no metadata score boost. The only post-retrieval filter is a hard
    season correctness filter (e.g. if the query mentions 2023, chunks
    from a 2024-only page are excluded).

    Input : query — natural-language question about IPL seasons, matches, or players.
    Output: top-k chunks ranked by cosine similarity with citations.
    """
    query = " ".join(str(query or "").split())
    if not query:
        return {"tool": "search_docs", "query": query, "results": [],
                "result_count": 0, "error": "Query cannot be empty."}

    top_k  = max(1, min(int(top_k), MAX_RESULTS))
    parsed = parse_query(query)

    if not _is_in_domain(query, parsed):
        return {"tool": "search_docs", "query": query, "results": [],
                "result_count": 0,
                "error": "Query appears to be outside the IPL document corpus."}

    try:
        index, chunks = _get_index()
    except Exception as exc:
        return {"tool": "search_docs", "query": query, "results": [],
                "result_count": 0, "error": f"Index error: {exc}"}

    # Embed query (normalised — cosine via inner product)
    q_vec = _embed([query])   # shape (1, dim)

    # Fetch extra candidates so deduplication has room to work
    fetch_k = min(top_k * 12, index.ntotal)
    raw_scores, raw_indices = index.search(q_vec, fetch_k)

    # Apply hard season filter only — no score manipulation
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for raw_score, idx in zip(raw_scores[0], raw_indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        # Hard season filter: wrong-season chunks are excluded entirely
        if parsed["season"] and chunk["season"] and chunk["season"] != parsed["season"]:
            continue
        candidates.append((float(raw_score), chunk))

    # Already sorted by FAISS (descending cosine similarity)
    # Build deduplicated results
    results: List[Dict[str, Any]] = []
    seen: set = set()
    for score, chunk in candidates:
        key = _dedup_key(chunk)
        if key in seen:
            continue
        seen.add(key)

        text      = chunk["text"]
        truncated = len(text) > MAX_CHARS_OUTPUT
        if truncated:
            text = text[:MAX_CHARS_OUTPUT].rstrip() + " [...]"

        results.append({
            "rank":       len(results) + 1,
            "text":       text,
            "source":     chunk["source"],
            "page":       int(chunk["page"]),
            "score":      round(score, 4),
            "season":     chunk.get("season"),
            "match_num":  chunk.get("match_num"),
            "is_final":   chunk.get("is_final"),
            "chunk_type": chunk.get("chunk_type"),
            "citation":   chunk["citation"],
        })
        if len(results) >= top_k:
            break

    if not results:
        return {"tool": "search_docs", "query": query, "results": [],
                "result_count": 0,
                "error": "No relevant document chunks found for this query."}

    return {
        "tool":         "search_docs",
        "query":        query,
        "results":      results,
        "result_count": len(results),
        "error":        None,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building / loading FAISS index...\n")
    _get_index()

    test_queries = [
        "Who won the IPL 2023 final?",
        "What was the impact of the new ball rule change in IPL 2024?",
        "Virat Kohli batting performance 2023",
        "Why did MI lose the 2023 final?",
        "Jasprit Bumrah bowling spell IPL 2024",
        "Premier League football",        # out of domain — should be refused
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        result = search_docs(q)
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            continue
        for r in result["results"]:
            print(f"\n  Rank {r['rank']} | cosine_score={r['score']} | {r['citation']} | {r['chunk_type']}")
            print(f"  {r['text'][:250]}...")

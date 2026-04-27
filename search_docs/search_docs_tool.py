"""
search_docs_tool.py  –  Vector-DB RAG for the IPL 2023/2024 PDF corpus.

Retrieval pipeline:
  1. PDF → pages → match-level chunks + sliding-window paragraph chunks
  2. Each chunk is embedded with sentence-transformers (all-MiniLM-L6-v2)
  3. Embeddings are stored in a FAISS IndexFlatIP (inner-product / cosine) index
  4. At query time the query is embedded and the top-K nearest chunks are
     retrieved, then re-ranked with a lightweight metadata boost.

The index (embeddings + chunk metadata) is persisted to disk so it is only
rebuilt when the PDF changes.
"""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Lazy imports for heavy ML deps so the module can be imported cheaply
# ---------------------------------------------------------------------------
def _import_faiss():
    import faiss  # noqa: PLC0415
    return faiss


def _import_sentence_transformer():
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    return SentenceTransformer


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PDF_NAME = "IPL_2023&2024_merged.pdf"
PDF_PATH = DATA_DIR / PDF_NAME
CACHE_PATH = DATA_DIR / "search_docs_index.pkl"

# Bump this whenever the chunking / embedding logic changes so the cache is
# automatically invalidated.
CACHE_VERSION = 10

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
MAX_RESULTS = 3
MAX_CHARS_PER_RESULT = 1800

TOOL_DESCRIPTION = (
    "Search the local IPL 2023 and IPL 2024 match-report PDF corpus. Use this tool "
    "for explanations, match-report context, player-performance narratives, playoff "
    "or final report details, and other facts written in the unstructured PDF. Do not "
    "use it for current/live news or for structured aggregate statistics that query_data "
    "can answer directly. Input is one natural-language query. Output is the top 3 text "
    "chunks with source filename, page number, score, and citation."
)

# ---------------------------------------------------------------------------
# Team / domain helpers (kept for metadata extraction & domain check)
# ---------------------------------------------------------------------------
TEAM_ALIASES = {
    "csk": ["Chennai Super Kings"],
    "mi": ["Mumbai Indians"],
    "rcb": ["Royal Challengers Bangalore", "Royal Challengers Bengaluru"],
    "kkr": ["Kolkata Knight Riders"],
    "rr": ["Rajasthan Royals"],
    "dc": ["Delhi Capitals"],
    "pbks": ["Punjab Kings"],
    "kxip": ["Punjab Kings"],
    "srh": ["Sunrisers Hyderabad"],
    "gt": ["Gujarat Titans"],
    "lsg": ["Lucknow Super Giants"],
}

FULL_TEAM_NAMES = [
    "chennai super kings",
    "mumbai indians",
    "royal challengers bangalore",
    "royal challengers bengaluru",
    "kolkata knight riders",
    "rajasthan royals",
    "delhi capitals",
    "punjab kings",
    "sunrisers hyderabad",
    "gujarat titans",
    "lucknow super giants",
]

ORDINAL_MAP = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10, "opener": 1, "opening": 1,
}

GENERIC_WORDS = {
    "Which", "Who", "What", "How", "When", "Where", "Match", "Game",
    "Fixture", "IPL", "Premier", "League", "Football", "Super", "Kings",
    "Royal", "Challengers", "Knight", "Riders", "Capitals", "Titans",
    "Giants", "Sunrisers",
}

DOMAIN_TERMS = {
    "ipl", "cricket", "match", "game", "fixture", "final", "qualifier",
    "eliminator", "playoff", "runs", "wickets", "batting", "bowling",
    "batter", "bowler", "over", "innings", "score", "chase", "impact player",
    "csk", "mi", "rcb", "kkr", "rr", "dc", "pbks", "srh", "gt", "lsg",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
MATCH_RE = re.compile(
    r"(?:^|\n)\s*(?:(2023|2024)\s*\n\s*)?((?:MATCH|Match)\s+\d+(?:\s+[A-Z0-9(). -]+)?)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _empty_response(query: str, error_message: Optional[str]) -> Dict[str, Any]:
    return {
        "tool": "search_docs",
        "query": query,
        "results": [],
        "result_count": 0,
        "error": error_message,
    }


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------
def _load_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF corpus: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = _clean_text(page.extract_text() or "")
        pages.append({"page": index, "text": text})
    return pages


def _page_offsets(pages: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int, int]]]:
    full_parts, offsets, pos = [], [], 0
    for page in pages:
        text = page["text"]
        full_parts.append(text)
        offsets.append((pos, pos + len(text), page["page"]))
        pos += len(text) + 1
    return "\n".join(full_parts), offsets


def _char_to_page(pos: int, offsets: List[Tuple[int, int, int]]) -> int:
    for start, end, page in offsets:
        if start <= pos < end:
            return page
    return offsets[-1][2] if offsets else 1


def _extract_teams(text: str) -> List[str]:
    text_norm = _norm(text)
    teams = []
    for full in FULL_TEAM_NAMES:
        if full in text_norm:
            teams.append(full.title())
    return list(dict.fromkeys(teams))


# ---------------------------------------------------------------------------
# Chunking  (same strategy as before – match-level + paragraph sliding window)
# ---------------------------------------------------------------------------
def _build_match_chunks(full_text: str, offsets: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    markers = list(MATCH_RE.finditer(full_text))
    current_season = 2023
    chunks = []

    def marker_text_start(match: re.Match) -> int:
        return match.start(1) if match.group(1) else match.start(2)

    for idx, marker in enumerate(markers):
        if marker.group(1):
            current_season = int(marker.group(1))
        label = marker.group(2).strip()
        match_number = re.search(r"\d+", label)
        if not match_number:
            continue
        start = marker_text_start(marker)
        end = marker_text_start(markers[idx + 1]) if idx + 1 < len(markers) else len(full_text)
        text = _clean_text(full_text[start:end])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        scorecard = lines[1] if len(lines) > 1 else (lines[0] if lines else "")
        chunks.append({
            "chunk_id": f"match-{current_season}-{int(match_number.group())}",
            "text": text,
            "source": PDF_NAME,
            "page": _char_to_page(start, offsets),
            "season": current_season,
            "match_num": int(match_number.group()),
            "match_label": label,
            "is_final": bool(re.search(r"\bfinal\b", label, re.I)),
            "is_playoff": bool(re.search(r"\b(qualifier|eliminator)\b", label, re.I)),
            "scorecard": scorecard,
            "teams": _extract_teams(text[:1000]),
            "chunk_type": "match",
        })
    return chunks


def _build_paragraph_chunks(
    match_chunks: List[Dict[str, Any]], window: int = 3, min_chars: int = 180
) -> List[Dict[str, Any]]:
    chunks = []
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    for match_chunk in match_chunks:
        sentences = sentence_re.split(match_chunk["text"])
        step = max(1, window - 1)
        for idx in range(0, len(sentences), step):
            para = " ".join(sentences[idx: idx + window]).strip()
            if len(para) < min_chars:
                continue
            chunks.append({
                **match_chunk,
                "chunk_id": f"{match_chunk['chunk_id']}-p{idx}",
                "text": para,
                "chunk_type": "paragraph",
            })
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed_texts(texts: List[str], model) -> np.ndarray:
    """Return L2-normalised float32 embeddings (shape: N × dim)."""
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Index build & load
# ---------------------------------------------------------------------------
def _build_index() -> Dict[str, Any]:
    """Build the FAISS vector index from scratch."""
    SentenceTransformer = _import_sentence_transformer()
    faiss = _import_faiss()

    model = SentenceTransformer(EMBEDDING_MODEL)

    pages = _load_pdf_pages(PDF_PATH)
    full_text, offsets = _page_offsets(pages)
    match_chunks = _build_match_chunks(full_text, offsets)
    paragraph_chunks = _build_paragraph_chunks(match_chunks)
    chunks = match_chunks + paragraph_chunks

    # Build embedding texts: chunk text + lightweight metadata prefix
    embed_texts = []
    for chunk in chunks:
        meta = " ".join([
            str(chunk.get("season", "")),
            f"match {chunk.get('match_num', '')}",
            chunk.get("match_label", ""),
            " ".join(chunk.get("teams", [])),
            chunk.get("scorecard", ""),
        ])
        embed_texts.append(chunk["text"] + " " + meta)

    embeddings = _embed_texts(embed_texts, model)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs = cosine
    faiss_index.add(embeddings)

    # Serialise FAISS index to bytes for pickling
    faiss_bytes = faiss.serialize_index(faiss_index)

    return {
        "cache_version": CACHE_VERSION,
        "pdf_mtime": PDF_PATH.stat().st_mtime,
        "chunks": chunks,
        "match_chunks": match_chunks,
        "faiss_bytes": faiss_bytes,
        "embedding_model": EMBEDDING_MODEL,
    }


def _load_or_build_index() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                cached.get("cache_version") == CACHE_VERSION
                and cached.get("pdf_mtime") == PDF_PATH.stat().st_mtime
                and cached.get("embedding_model") == EMBEDDING_MODEL
            ):
                return cached
        except Exception:
            pass

    index = _build_index()
    try:
        with CACHE_PATH.open("wb") as handle:
            pickle.dump(index, handle)
    except Exception:
        pass
    return index


@lru_cache(maxsize=1)
def _index() -> Dict[str, Any]:
    return _load_or_build_index()


def _get_faiss_index(state: Dict[str, Any]):
    """Deserialise the FAISS index from the cached bytes."""
    faiss = _import_faiss()
    return faiss.deserialize_index(state["faiss_bytes"])


# ---------------------------------------------------------------------------
# Query parsing  (unchanged – used for metadata boost & domain check)
# ---------------------------------------------------------------------------
def parse_query(query: str) -> Dict[str, Any]:
    q = _norm(query)
    season_match = re.search(r"\b(2023|2024)\b", q)
    season = int(season_match.group(1)) if season_match else None

    match_num = None
    explicit_match = re.search(r"\b(?:match|game|fixture)\s+(\d+)\b", q)
    if explicit_match:
        match_num = int(explicit_match.group(1))

    if match_num is None:
        for word, number in ORDINAL_MAP.items():
            if re.search(r"\b" + re.escape(word) + r"\b", q) and re.search(
                r"\b(match|game|fixture|ipl|season|opener|opening)\b", q
            ):
                match_num = number
                break

    teams = []
    for alias, full_names in TEAM_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            teams.extend(full_names)
    for name in FULL_TEAM_NAMES:
        if name in q:
            teams.append(name.title())
    teams = list(dict.fromkeys(teams))

    player_candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    players = [
        p for p in player_candidates
        if not any(part in GENERIC_WORDS for part in p.split())
    ]

    return {
        "season": season,
        "match_num": match_num,
        "teams": teams,
        "players": players,
        "is_final": bool(re.search(r"\bfinal\b", q)),
        "is_playoff": bool(re.search(r"\b(qualifier|eliminator|playoff)\b", q)),
        "wants_bowling": bool(re.search(
            r"\b(bowling|bowler|spell|wicket|wickets|dismissed|figures)\b", q
        )),
        "wants_batting": bool(re.search(
            r"\b(batting|batter|batting|runs|score|scored|century|fifty|finish)\b", q
        )),
    }


def _has_domain_term(q: str) -> bool:
    for term in DOMAIN_TERMS:
        if " " in term:
            if term in q:
                return True
        elif re.search(r"\b" + re.escape(term) + r"\b", q):
            return True
    return False


def _is_in_domain(query: str, parsed: Dict[str, Any]) -> bool:
    q = _norm(query)
    if parsed["season"] or parsed["match_num"] or parsed["teams"] or parsed["players"]:
        return True
    if _has_domain_term(q):
        return True
    if any(team in q for team in FULL_TEAM_NAMES):
        return True
    return False


# ---------------------------------------------------------------------------
# Metadata boost  (applied after vector retrieval for re-ranking)
# ---------------------------------------------------------------------------
def _metadata_boost(chunk: Dict[str, Any], parsed: Dict[str, Any]) -> float:
    boost = 0.0
    if parsed["match_num"] and chunk.get("match_num") == parsed["match_num"]:
        boost += 4.0
    if parsed["is_final"] and chunk.get("is_final"):
        boost += 4.0
    if parsed["is_playoff"] and chunk.get("is_playoff"):
        boost += 1.5

    chunk_text = _norm(chunk["text"])
    for team in parsed["teams"]:
        if _norm(team) in chunk_text:
            boost += 1.0
    for player in parsed["players"]:
        player_norm = _norm(player)
        player_terms = [part for part in player_norm.split() if len(part) > 3]
        if player_norm and player_norm in chunk_text:
            boost += 12.0
        term_mentions = sum(chunk_text.count(part) for part in player_terms)
        if term_mentions:
            boost += min(18.0, 5.0 * term_mentions)

    if parsed.get("wants_bowling") and re.search(
        r"\b(wicket|wickets|bowled|bowling|spell|figures)\b", chunk_text
    ):
        boost += 8.0
    if parsed.get("wants_batting") and re.search(
        r"\b(run|runs|batting|batter|scored|score|century|fifty|finish|chase)\b", chunk_text
    ):
        boost += 6.0
    return boost


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def _format_result(
    chunk: Dict[str, Any], rank: int, score: float, reason: str
) -> Dict[str, Any]:
    text = chunk["text"]
    truncated = len(text) > MAX_CHARS_PER_RESULT
    if truncated:
        text = text[:MAX_CHARS_PER_RESULT].rstrip() + " [...]"
    return {
        "rank": rank,
        "text": text,
        "source": chunk["source"],
        "page": int(chunk["page"]),
        "score": round(float(score), 4),
        "season": chunk.get("season"),
        "match_num": chunk.get("match_num"),
        "chunk_type": chunk.get("chunk_type"),
        "citation": f"{chunk['source']}, p.{chunk['page']}",
        "reason": reason,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Direct metadata lookup  (exact match shortcut – unchanged)
# ---------------------------------------------------------------------------
def _direct_candidates(
    match_chunks: List[Dict[str, Any]], parsed: Dict[str, Any]
) -> List[Tuple[float, Dict[str, Any], str]]:
    candidates = []
    for chunk in match_chunks:
        if parsed["season"] and chunk["season"] != parsed["season"]:
            continue
        if parsed["match_num"] and chunk["match_num"] == parsed["match_num"]:
            candidates.append((100.0, chunk, "direct season/match metadata lookup"))
        elif parsed["is_final"] and chunk["is_final"]:
            candidates.append((95.0, chunk, "direct final metadata lookup"))
        elif parsed["is_playoff"] and chunk["is_playoff"]:
            candidates.append((80.0, chunk, "direct playoff metadata lookup"))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def search_docs(query: str, top_k: int = MAX_RESULTS) -> Dict[str, Any]:
    """
    Retrieve the top-k most relevant chunks from the IPL PDF corpus using
    vector similarity search (FAISS + sentence-transformers) with a
    metadata re-ranking boost.
    """
    query = " ".join(str(query or "").split())
    if not query:
        return _empty_response(query, "Query cannot be empty.")

    top_k = max(1, min(int(top_k), MAX_RESULTS))
    parsed = parse_query(query)

    if not _is_in_domain(query, parsed):
        return _empty_response(query, "Query appears outside the IPL document corpus.")

    try:
        state = _index()
    except Exception as exc:
        return _empty_response(query, f"Document index could not be loaded: {exc}")

    chunks = state["chunks"]
    match_chunks = state["match_chunks"]

    # ------------------------------------------------------------------
    # 1. Fast-path: exact metadata match (season + match number / final)
    # ------------------------------------------------------------------
    direct = _direct_candidates(match_chunks, parsed)
    results: List[Dict[str, Any]] = []
    seen_matches: set = set()

    for score, chunk, reason in direct:
        key = (chunk["season"], chunk["match_num"])
        if key in seen_matches:
            continue
        seen_matches.add(key)
        results.append(_format_result(chunk, len(results) + 1, score, reason))
        if len(results) >= top_k:
            return {
                "tool": "search_docs",
                "query": query,
                "results": results,
                "result_count": len(results),
                "error": None,
            }

    # ------------------------------------------------------------------
    # 2. Vector retrieval via FAISS
    # ------------------------------------------------------------------
    SentenceTransformer = _import_sentence_transformer()
    model = SentenceTransformer(EMBEDDING_MODEL)
    faiss_index = _get_faiss_index(state)

    # Embed the query (normalised so inner product = cosine similarity)
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Retrieve more candidates than needed so re-ranking has room to work
    retrieve_k = min(len(chunks), max(top_k * 10, 30))
    cosine_scores, indices = faiss_index.search(query_vec, retrieve_k)
    cosine_scores = cosine_scores[0]   # shape: (retrieve_k,)
    indices = indices[0]

    # Season filter + metadata boost + re-rank
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for cos_score, idx in zip(cosine_scores, indices):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        if parsed["season"] and chunk.get("season") != parsed["season"]:
            continue
        final_score = float(cos_score) + _metadata_boost(chunk, parsed) * 0.05
        scored.append((final_score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    for score, chunk in scored:
        key = (chunk.get("season"), chunk.get("match_num"))
        if key in seen_matches:
            continue
        seen_matches.add(key)
        results.append(
            _format_result(chunk, len(results) + 1, score, "vector similarity + metadata boost")
        )
        if len(results) >= top_k:
            break

    if not results:
        return _empty_response(query, "No relevant document chunks found.")

    return {
        "tool": "search_docs",
        "query": query,
        "results": results,
        "result_count": len(results),
        "error": None,
    }

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PDF_NAME = "IPL_2023&2024_merged.pdf"
PDF_PATH = DATA_DIR / PDF_NAME
CACHE_PATH = DATA_DIR / "search_docs_index.pkl"
CACHE_VERSION = 5

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
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
    "opener": 1,
    "opening": 1,
}

GENERIC_WORDS = {
    "Which",
    "Who",
    "What",
    "How",
    "When",
    "Where",
    "Match",
    "Game",
    "Fixture",
    "IPL",
    "Premier",
    "League",
    "Football",
    "Super",
    "Kings",
    "Royal",
    "Challengers",
    "Knight",
    "Riders",
    "Capitals",
    "Titans",
    "Giants",
    "Sunrisers",
}

DOMAIN_TERMS = {
    "ipl",
    "cricket",
    "match",
    "game",
    "fixture",
    "final",
    "qualifier",
    "eliminator",
    "playoff",
    "runs",
    "wickets",
    "batting",
    "bowling",
    "batter",
    "bowler",
    "over",
    "innings",
    "score",
    "chase",
    "impact player",
    "csk",
    "mi",
    "rcb",
    "kkr",
    "rr",
    "dc",
    "pbks",
    "srh",
    "gt",
    "lsg",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

QUERY_SYNONYMS = [
    (r"\bfirst\b", "match 1"),
    (r"\bsecond\b", "match 2"),
    (r"\bthird\b", "match 3"),
    (r"\bopening match\b", "match 1"),
    (r"\bopener\b", "match 1"),
    (r"\bgame\s+(\d+)\b", r"match \1"),
    (r"\bfixture\s+(\d+)\b", r"match \1"),
    (r"\bdefeated\b", "beat"),
    (r"\bvictorious\b", "won"),
    (r"\bvictory\b", "win"),
    (r"\bchampion\b", "won final"),
    (r"\btitle winner\b", "won final"),
    (r"\bscored\b", "runs"),
    (r"\bhit\b", "runs"),
    (r"\bdismissed\b", "wicket"),
    (r"\btook wickets\b", "bowling wickets"),
    (r"\bpicked up\b", "took wickets"),
    (r"\bcsk\b", "chennai super kings"),
    (r"\bmi\b", "mumbai indians"),
    (r"\brcb\b", "royal challengers"),
    (r"\bkkr\b", "kolkata knight riders"),
    (r"\brr\b", "rajasthan royals"),
    (r"\bdc\b", "delhi capitals"),
    (r"\bpbks\b", "punjab kings"),
    (r"\bsrh\b", "sunrisers hyderabad"),
    (r"\bgt\b", "gujarat titans"),
    (r"\blsg\b", "lucknow super giants"),
]

TOKEN_RE = re.compile(r"[a-z0-9]+")
MATCH_RE = re.compile(
    r"(?:^|\n)\s*(?:(2023|2024)\s*\n\s*)?((?:MATCH|Match)\s+\d+(?:\s+[A-Z0-9(). -]+)?)",
    re.MULTILINE,
)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS and len(tok) > 1]


def _expand_query(query: str) -> str:
    expanded = query.lower()
    for pattern, replacement in QUERY_SYNONYMS:
        expanded = re.sub(pattern, replacement, expanded)
    return f"{query} {expanded}"


def _empty_response(query: str, error_message: Optional[str]) -> Dict[str, Any]:
    return {
        "tool": "search_docs",
        "query": query,
        "results": [],
        "result_count": 0,
        "error": error_message,
    }


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
    full_parts = []
    offsets = []
    pos = 0
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

        chunks.append(
            {
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
            }
        )
    return chunks


def _build_paragraph_chunks(match_chunks: List[Dict[str, Any]], window: int = 3, min_chars: int = 180) -> List[Dict[str, Any]]:
    chunks = []
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    for match_chunk in match_chunks:
        sentences = sentence_re.split(match_chunk["text"])
        step = max(1, window - 1)
        for idx in range(0, len(sentences), step):
            para = " ".join(sentences[idx : idx + window]).strip()
            if len(para) < min_chars:
                continue
            chunks.append(
                {
                    **match_chunk,
                    "chunk_id": f"{match_chunk['chunk_id']}-p{idx}",
                    "text": para,
                    "chunk_type": "paragraph",
                }
            )
    return chunks


def _build_index() -> Dict[str, Any]:
    pages = _load_pdf_pages(PDF_PATH)
    full_text, offsets = _page_offsets(pages)
    match_chunks = _build_match_chunks(full_text, offsets)
    paragraph_chunks = _build_paragraph_chunks(match_chunks)
    chunks = match_chunks + paragraph_chunks

    token_lists = []
    counters = []
    doc_freq = Counter()
    for chunk in chunks:
        metadata_text = " ".join(
            [
                str(chunk.get("season", "")),
                f"match {chunk.get('match_num', '')}",
                chunk.get("match_label", ""),
                " ".join(chunk.get("teams", [])),
                chunk.get("scorecard", ""),
            ]
        )
        toks = _tokens(chunk["text"] + " " + metadata_text)
        token_lists.append(toks)
        counter = Counter(toks)
        counters.append(counter)
        doc_freq.update(counter.keys())

    total_docs = len(chunks)
    avgdl = sum(len(toks) for toks in token_lists) / max(total_docs, 1)
    idf = {
        term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    return {
        "cache_version": CACHE_VERSION,
        "pdf_mtime": PDF_PATH.stat().st_mtime,
        "chunks": chunks,
        "match_chunks": match_chunks,
        "counters": counters,
        "doc_lengths": [len(toks) for toks in token_lists],
        "avgdl": avgdl,
        "idf": idf,
    }


def _load_or_build_index() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("cache_version") == CACHE_VERSION and cached.get("pdf_mtime") == PDF_PATH.stat().st_mtime:
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
            if re.search(r"\b" + re.escape(word) + r"\b", q) and re.search(r"\b(match|game|fixture|ipl|season|opener|opening)\b", q):
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
        player
        for player in player_candidates
        if not any(part in GENERIC_WORDS for part in player.split())
    ]

    return {
        "season": season,
        "match_num": match_num,
        "teams": teams,
        "players": players,
        "is_final": bool(re.search(r"\bfinal\b", q)),
        "is_playoff": bool(re.search(r"\b(qualifier|eliminator|playoff)\b", q)),
        "wants_bowling": bool(re.search(r"\b(bowling|bowler|spell|wicket|wickets|dismissed|figures)\b", q)),
        "wants_batting": bool(re.search(r"\b(batting|batter|batting|runs|score|scored|century|fifty|finish)\b", q)),
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


def _bm25_score(query_terms: List[str], counter: Counter, doc_len: int, avgdl: float, idf: Dict[str, float]) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term in query_terms:
        freq = counter.get(term, 0)
        if not freq:
            continue
        denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
        score += idf.get(term, 0.0) * (freq * (k1 + 1)) / denom
    return score


def _chunk_mentions_named_entities(chunk: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
    chunk_text = _norm(chunk["text"])

    for player in parsed["players"]:
        player_terms = [part for part in _norm(player).split() if len(part) > 3]
        if player_terms and not any(part in chunk_text for part in player_terms):
            return False

    for team in parsed["teams"]:
        team_terms = [part for part in _norm(team).split() if len(part) > 3]
        if team_terms and not any(part in chunk_text for part in team_terms):
            return False

    return True


def _metadata_boost(chunk: Dict[str, Any], parsed: Dict[str, Any]) -> float:
    boost = 0.0
    if parsed["match_num"] and chunk["match_num"] == parsed["match_num"]:
        boost += 4.0
    if parsed["is_final"] and chunk["is_final"]:
        boost += 4.0
    if parsed["is_playoff"] and chunk["is_playoff"]:
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

    if parsed.get("wants_bowling") and re.search(r"\b(wicket|wickets|bowled|bowling|spell|figures|double wicket|four wicket)\b", chunk_text):
        boost += 8.0
    if parsed.get("wants_batting") and re.search(r"\b(run|runs|batting|batter|scored|score|century|fifty|finish|chase)\b", chunk_text):
        boost += 6.0
    return boost


def _format_result(chunk: Dict[str, Any], rank: int, score: float, reason: str) -> Dict[str, Any]:
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


def _direct_candidates(match_chunks: List[Dict[str, Any]], parsed: Dict[str, Any]) -> List[Tuple[float, Dict[str, Any], str]]:
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


def search_docs(query: str, top_k: int = MAX_RESULTS) -> Dict[str, Any]:
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
    direct = _direct_candidates(match_chunks, parsed)

    results: List[Dict[str, Any]] = []
    seen_matches = set()

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

    expanded_query = _expand_query(query)
    query_terms = _tokens(expanded_query)
    if not query_terms:
        return _empty_response(query, "Query did not contain searchable terms.")

    scored = []
    for idx, chunk in enumerate(chunks):
        if parsed["season"] and chunk["season"] != parsed["season"]:
            continue
        if not _chunk_mentions_named_entities(chunk, parsed):
            continue
        base = _bm25_score(
            query_terms,
            state["counters"][idx],
            state["doc_lengths"][idx],
            state["avgdl"],
            state["idf"],
        )
        score = base + _metadata_boost(chunk, parsed)
        if score <= 0:
            continue
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    for score, chunk in scored:
        key = (chunk["season"], chunk["match_num"])
        if key in seen_matches:
            continue
        seen_matches.add(key)
        results.append(_format_result(chunk, len(results) + 1, score, "BM25 retrieval with metadata boosts"))
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


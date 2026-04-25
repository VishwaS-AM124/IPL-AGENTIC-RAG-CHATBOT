from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


TAVILY_ENDPOINT = "https://api.tavily.com/search"
MAX_QUERY_WORDS = 10
MAX_RESULTS = 3

TOOL_DESCRIPTION = (
    "Search the live web for recent or current IPL information that is not available "
    "in the local structured dataset or document corpus. Use this tool for current "
    "news, current squads, recent injuries, live/current standings, and fresh public "
    "facts. The input must be a short search query of 10 words or fewer. Do not use "
    "this tool for historical match statistics that query_data can answer."
)


def _load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """Tiny .env loader so the tool does not require python-dotenv."""
    path = dotenv_path or Path(__file__).resolve().parent / ".env"
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


def _empty_response(query: str, error_message: Optional[str] = None) -> Dict[str, Any]:
    return {
        "tool": "web_search",
        "query": query,
        "results": [],
        "result_count": 0,
        "error": error_message,
    }


def _clean_text(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _published_date(item: Dict[str, Any]) -> Optional[str]:
    for key in ("published_date", "publishedDate", "date"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def web_search(query: str, *, max_results: int = MAX_RESULTS, timeout: int = 15) -> Dict[str, Any]:
    """Run a Tavily web search and return top snippets with URL/date citations.

    Input:
        query: short search query, 10 words or fewer.

    Output:
        {
            "tool": "web_search",
            "query": str,
            "results": [
                {
                    "title": str,
                    "snippet": str,
                    "url": str,
                    "published_date": str | None,
                    "citation": str
                }
            ],
            "result_count": int,
            "error": str | None
        }
    """
    query = " ".join(str(query or "").split())
    if not query:
        return _empty_response(query, "Query cannot be empty.")

    if len(query.split()) > MAX_QUERY_WORDS:
        return _empty_response(query, f"Query must be {MAX_QUERY_WORDS} words or fewer.")

    if max_results < 1:
        return _empty_response(query, "max_results must be at least 1.")

    max_results = min(max_results, MAX_RESULTS)

    _load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return _empty_response(
            query,
            "Missing TAVILY_API_KEY. Add it to your environment or to a local .env file.",
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        data = _post_json(TAVILY_ENDPOINT, payload, timeout)
    except TimeoutError:
        return _empty_response(query, "Web search timed out.")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return _empty_response(query, f"Web search HTTP error {exc.code}: {detail}")
    except error.URLError as exc:
        return _empty_response(query, f"Web search request failed: {exc.reason}")
    except json.JSONDecodeError:
        return _empty_response(query, "Web search returned invalid JSON.")

    if data.get("error"):
        return _empty_response(query, f"Tavily error: {data['error']}")

    results: List[Dict[str, Any]] = []
    for item in data.get("results", [])[:max_results]:
        url_value = str(item.get("url", "")).strip()
        date = _published_date(item)
        results.append(
            {
                "title": _clean_text(item.get("title"), limit=180),
                "snippet": _clean_text(item.get("content") or item.get("snippet"), limit=500),
                "url": url_value,
                "published_date": date,
                "citation": f"{url_value} ({date or 'publication date unavailable'})",
            }
        )

    if not results:
        return _empty_response(query, "No results found.")

    return {
        "tool": "web_search",
        "query": query,
        "results": results,
        "result_count": len(results),
        "error": None,
    }

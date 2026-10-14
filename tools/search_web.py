"""
tools/search_web.py
Core web search primitive for MAVIS using DuckDuckGo (via ddgs / duckduckgo-search).
Does not require an external API key.
"""
from __future__ import annotations

import time

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> tuple[int, str]:
    """
    Searches the web for information based on a given query using DuckDuckGo Search.

    Args:
        query (str): The search query for web results.
        max_results (int, optional): Maximum number of search results to retrieve (default: 5).

    Returns:
        tuple[int, str]: A tuple containing:
                          - status_code (int): 0 for success, -1 for failure.
                          - result (str): A formatted string of search results on success,
                                         or an error message string on failure.
    """
    if not query or not str(query).strip():
        return 0, "No results found for empty query."

    query_str = str(query).strip()
    try:
        limit = int(max_results) if max_results and int(max_results) > 0 else 5
    except (ValueError, TypeError):
        limit = 5

    last_error = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query_str, max_results=limit))

            if not raw_results:
                return 0, f"No results found for query: '{query_str}'"

            formatted_results = []
            for i, item in enumerate(raw_results):
                title = item.get("title", "N/A")
                snippet = item.get("body", item.get("snippet", "N/A"))
                link = item.get("href", item.get("url", item.get("link", "N/A")))
                formatted_results.append(
                    f"Result {i + 1}:\nTitle: {title}\nSnippet: {snippet}\nLink: {link}"
                )

            return 0, "\n\n".join(formatted_results)
        except Exception as e:
            err_msg = str(e).strip()
            if "no results" in err_msg.lower():
                return 0, f"No results found for query: '{query_str}'"
            last_error = e
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

    if last_error and "no results" in str(last_error).lower():
        return 0, f"No results found for query: '{query_str}'"

    return -1, f"An unexpected error occurred during web search: {last_error}"

"""Tavily web search tool."""
from __future__ import annotations

import os
from langchain_core.tools import tool


def make_web_search_tool():
    """Create the search tool only when TAVILY_API_KEY is configured."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to .env to enable web search."
        )
    try:
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=5, topic="general")
    except ImportError as exc:
        raise RuntimeError(
            "langchain-tavily is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    @tool
    def web_search_tool(query: str) -> str:
        """Search the web for general knowledge, Bangladesh policies, official
        guidance, definitions and current context not contained in the datasets.
        Prefer official government or primary sources when available. Summarize
        search results and retain source URLs in the output."""
        result = search.invoke({"query": query})
        return str(result)

    return web_search_tool

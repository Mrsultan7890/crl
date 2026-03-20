"""
Search engine integration — free URL discovery via DuckDuckGo.

No API key required. Uses duckduckgo-search (pure Python).

Usage:
    from crl.search import search_urls, search_and_crawl

    urls = search_urls("python async programming", max_results=10)
    results = search_and_crawl("python async programming", top_k=5)
"""

import asyncio
import logging
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

SearchBackend = Literal["ddg"]


def search_urls(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
    safesearch: str = "off",
    timelimit: Optional[str] = None,
) -> List[str]:
    """
    Search DuckDuckGo and return a list of result URLs.

    Free, no API key needed.

    Args:
        query: Search query string.
        max_results: Max URLs to return (default 10).
        region: DDG region code e.g. 'us-en', 'wt-wt' (worldwide).
        safesearch: 'on', 'moderate', or 'off'.
        timelimit: 'd' (day), 'w' (week), 'm' (month), 'y' (year), or None.

    Returns:
        List of URLs from search results.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "duckduckgo-search not installed. Run: pip install duckduckgo-search"
        )

    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results,
        ):
            if r.get("href"):
                urls.append(r["href"])
    logger.info("DDG search '%s' → %d URLs", query, len(urls))
    return urls


def search_news_urls(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
    timelimit: Optional[str] = "w",
) -> List[str]:
    """
    Search DuckDuckGo News and return URLs.

    Args:
        query: News search query.
        max_results: Max URLs to return.
        region: DDG region code.
        timelimit: 'd', 'w', 'm', or None.

    Returns:
        List of news article URLs.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "duckduckgo-search not installed. Run: pip install duckduckgo-search"
        )

    urls = []
    with DDGS() as ddgs:
        for r in ddgs.news(
            query,
            region=region,
            timelimit=timelimit,
            max_results=max_results,
        ):
            if r.get("url"):
                urls.append(r["url"])
    logger.info("DDG news '%s' → %d URLs", query, len(urls))
    return urls


def search_and_crawl(
    query: str,
    max_results: int = 10,
    top_k: Optional[int] = None,
    mode: str = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
) -> List[dict]:
    """
    Search DuckDuckGo for query, crawl results, rank by relevance.

    One-liner: query → URLs → crawl → rank.

    Args:
        query: Search query (used for both DDG search and relevance ranking).
        max_results: Max DDG results to fetch URLs from.
        top_k: Return only top K ranked results.
        mode: Scoring mode — 'keyword', 'semantic', or 'both'.
        semantic_weight: Semantic weight in 'both' mode.
        timeout: Per-request timeout.
        max_connections: Max concurrent connections.
        retries: Retry attempts per URL.
        rate_limit: Max requests per second.
        proxies: Proxy list for rotation.
        cache: TieredCache instance.
        respect_robots: Respect robots.txt.
        progress: ProgressReporter instance.
        min_text_length: Skip pages shorter than this.
        model_name: Sentence-transformers model.
        region: DDG region code.
        timelimit: DDG time filter.

    Returns:
        List of dicts sorted by relevance_score descending.
    """
    from . import crawl as _crawl

    urls = search_urls(query, max_results=max_results, region=region, timelimit=timelimit)
    if not urls:
        logger.warning("No URLs found for query: %s", query)
        return []

    return _crawl(
        urls=urls,
        query=query,
        top_k=top_k,
        mode=mode,
        semantic_weight=semantic_weight,
        timeout=timeout,
        max_connections=max_connections,
        retries=retries,
        rate_limit=rate_limit,
        proxies=proxies,
        cache=cache,
        respect_robots=respect_robots,
        progress=progress,
        min_text_length=min_text_length,
        model_name=model_name,
    )


async def asearch_and_crawl(
    query: str,
    max_results: int = 10,
    top_k: Optional[int] = None,
    mode: str = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
) -> List[dict]:
    """Async version of search_and_crawl()."""
    from . import acrawl

    urls = await asyncio.get_event_loop().run_in_executor(
        None, lambda: search_urls(query, max_results=max_results, region=region, timelimit=timelimit)
    )
    if not urls:
        return []

    return await acrawl(
        urls=urls,
        query=query,
        top_k=top_k,
        mode=mode,
        semantic_weight=semantic_weight,
        timeout=timeout,
        max_connections=max_connections,
        retries=retries,
        rate_limit=rate_limit,
        proxies=proxies,
        cache=cache,
        respect_robots=respect_robots,
        progress=progress,
        min_text_length=min_text_length,
        model_name=model_name,
    )

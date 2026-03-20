"""
CRL — Crawl Relevance Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure Python async web crawling with BM25 + semantic relevance ranking.

Basic usage::

    from crl import crawl

    results = crawl(
        urls=["https://example.com", "https://python.org"],
        query="python async programming",
        top_k=5,
        mode="both",
    )
    for r in results:
        print(r["url"], r["relevance_score"])

Async usage::

    import asyncio
    from crl import acrawl

    results = asyncio.run(acrawl(urls=[...], query="..."))

Streaming usage::

    import asyncio
    from crl import astream

    async def main():
        async for result in astream(urls=[...], query="..."):
            print(result["url"], result["relevance_score"])

    asyncio.run(main())

Search + crawl::

    from crl import search_and_crawl

    results = search_and_crawl("python async programming", max_results=10)
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Literal, Optional

from .fetcher import fetch, fetch_all
from .output import save, to_csv, to_dict, to_json, to_text
from .parser import parse, parse_many
from .relevance import ModeType, keyword_score, rank, semantic_score
from .crawler import deep_crawl, deep_crawl_sync
from .deduplicator import Deduplicator
from .paginator import next_page_url, build_page_urls, detect_pagination_links
from .cache import TieredCache, MemoryCache, DiskCache
from .progress import ProgressReporter, CrawlEvent
from .robots import RobotsCache
from .bridge import ZeroCopyTokenizer, FastHTMLStripper, MMapStore, tokenize, strip_html, make_store
from .sitemap import fetch_sitemap_urls, fetch_sitemap_urls_sync
from .search import search_urls, search_news_urls, search_and_crawl, asearch_and_crawl
from .ratelimiter import DomainRateLimiter
from .extractor import extract as extract_structured
from .js_renderer import render_sync, js_crawl, js_crawl_sync, is_available as js_available
from .output import to_markdown, to_sqlite

__version__ = "1.1.1"
__all__ = [
    "crawl",
    "acrawl",
    "astream",
    "deep_crawl",
    "deep_crawl_sync",
    "fetch",
    "fetch_all",
    "parse",
    "parse_many",
    "rank",
    "keyword_score",
    "semantic_score",
    "to_json",
    "to_dict",
    "to_text",
    "to_csv",
    "save",
    "Deduplicator",
    "next_page_url",
    "build_page_urls",
    "detect_pagination_links",
    "deep_search",
    "adeep_search",
    "TieredCache",
    "MemoryCache",
    "DiskCache",
    "ProgressReporter",
    "CrawlEvent",
    "RobotsCache",
    "ZeroCopyTokenizer",
    "FastHTMLStripper",
    "MMapStore",
    "tokenize",
    "strip_html",
    "make_store",
    "fetch_sitemap_urls",
    "fetch_sitemap_urls_sync",
    "search_urls",
    "search_news_urls",
    "search_and_crawl",
    "asearch_and_crawl",
    "DomainRateLimiter",
    "extract_structured",
    "render_sync",
    "js_crawl",
    "js_crawl_sync",
    "js_available",
    "to_markdown",
    "to_sqlite",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


def crawl(
    urls: List[str],
    query: str,
    top_k: Optional[int] = None,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[dict]:
    """
    Crawl URLs and return pages ranked by relevance to query.

    Args:
        urls: List of URLs to crawl.
        query: Search query to rank results against.
        top_k: Return only top K results. None returns all.
        mode: Scoring mode — 'keyword', 'semantic', or 'both'.
        semantic_weight: Weight for semantic score in 'both' mode (0.0–1.0).
        timeout: Per-request timeout in seconds.
        max_connections: Max concurrent HTTP connections.
        retries: Number of retry attempts per URL on failure.
        backoff: Exponential backoff multiplier between retries.
        rate_limit: Max requests per second (None = unlimited).
        min_text_length: Skip pages with fewer extracted characters.
        model_name: Sentence-transformers model for semantic scoring.

    Returns:
        List of dicts sorted by relevance_score descending.
    """
    raw = fetch(urls, timeout=timeout, max_connections=max_connections,
                retries=retries, backoff=backoff, rate_limit=rate_limit,
                proxies=proxies, cache=cache, respect_robots=respect_robots,
                progress=progress)
    parsed = parse_many(raw, min_text_length=min_text_length)
    return rank(parsed, query, top_k=top_k, mode=mode,
                semantic_weight=semantic_weight, model_name=model_name)


async def acrawl(
    urls: List[str],
    query: str,
    top_k: Optional[int] = None,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[dict]:
    """
    Async version of crawl(). Use inside async contexts.

    Example::

        import asyncio
        results = asyncio.run(acrawl(urls=[...], query="..."))
    """
    raw = await fetch_all(urls, timeout=timeout, max_connections=max_connections,
                          retries=retries, backoff=backoff, rate_limit=rate_limit,
                          proxies=proxies, cache=cache, respect_robots=respect_robots,
                          progress=progress)
    parsed = parse_many(raw, min_text_length=min_text_length)
    return rank(parsed, query, top_k=top_k, mode=mode,
                semantic_weight=semantic_weight, model_name=model_name)


async def astream(
    urls: List[str],
    query: str,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> AsyncIterator[dict]:
    """
    Async generator — yields each result as soon as it's fetched and scored.

    No waiting for all URLs to finish. Real-time streaming.

    Example::

        async for result in astream(urls=[...], query="python"):
            print(result["url"], result["relevance_score"])
    """
    from .fetcher import fetch_one, _DEFAULT_HEADERS, _validate_urls
    from .robots import RobotsCache
    import httpx

    _validate_urls(urls)
    robots = RobotsCache() if respect_robots else None

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections // 2,
    )
    async with httpx.AsyncClient(
        limits=limits, headers=_DEFAULT_HEADERS,
        http2=True, follow_redirects=True, verify=False,
    ) as client:
        tasks = {
            asyncio.ensure_future(
                fetch_one(client, url, timeout, retries, backoff,
                          cache=cache, robots=robots)
            ): url
            for url in urls
        }
        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                raw = fut.result()
                if raw.get("error") or not raw.get("html"):
                    continue
                parsed = parse_many([raw], min_text_length=min_text_length)
                if not parsed:
                    continue
                ranked = rank(parsed, query, mode=mode,
                              semantic_weight=semantic_weight, model_name=model_name)
                if ranked:
                    yield ranked[0]


def deep_search(
    urls: List[str],
    query: str,
    depth: int = 2,
    max_pages: int = 100,
    max_pages_per_domain: int = 50,
    follow_external: bool = False,
    paginate: bool = True,
    max_pagination_pages: int = 5,
    top_k: Optional[int] = None,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    domain_rate_limits: Optional[Dict[str, float]] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    similarity_threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    use_mmap: bool = False,
    mmap_max_mb: int = 256,
) -> List[dict]:
    """
    Full deep search: BFS depth crawl + pagination + dedup + relevance ranking.

    Args:
        urls: Seed URLs (e.g. search result URLs).
        query: Query to rank all collected pages against.
        depth: Link-follow depth (0 = seed only, 2 = recommended).
        max_pages: Hard cap on total pages crawled.
        max_pages_per_domain: Per-domain page cap.
        follow_external: Follow links to other domains.
        paginate: Auto-follow pagination (next page links).
        max_pagination_pages: Max pagination pages per seed URL.
        top_k: Return only top K ranked results.
        mode: Scoring mode — 'keyword', 'semantic', or 'both'.
        semantic_weight: Semantic score weight in 'both' mode (0.0–1.0).
        timeout: Per-request timeout in seconds.
        max_connections: Max concurrent HTTP connections.
        retries: Retry attempts per URL.
        backoff: Exponential backoff multiplier.
        rate_limit: Global max requests per second.
        domain_rate_limits: Per-domain RPS overrides e.g. {'example.com': 2.0}.
        min_text_length: Skip pages shorter than this.
        similarity_threshold: Content dedup threshold (0–1).
        model_name: Sentence-transformers model name.
        use_mmap: Use MMapStore for memory-efficient result storage.
        mmap_max_mb: MMapStore max size in MB.

    Returns:
        List of dicts sorted by relevance_score descending.
    """
    pages = deep_crawl_sync(
        urls=urls, depth=depth, max_pages=max_pages,
        max_pages_per_domain=max_pages_per_domain,
        follow_external=follow_external, paginate=paginate,
        max_pagination_pages=max_pagination_pages, timeout=timeout,
        max_connections=max_connections, retries=retries, backoff=backoff,
        rate_limit=rate_limit, domain_rate_limits=domain_rate_limits,
        proxies=proxies, cache=cache,
        respect_robots=respect_robots, progress=progress,
        min_text_length=min_text_length, similarity_threshold=similarity_threshold,
        use_mmap=use_mmap, mmap_max_mb=mmap_max_mb,
    )
    return rank(pages, query, top_k=top_k, mode=mode,
                semantic_weight=semantic_weight, model_name=model_name)


async def adeep_search(
    urls: List[str],
    query: str,
    depth: int = 2,
    max_pages: int = 100,
    max_pages_per_domain: int = 50,
    follow_external: bool = False,
    paginate: bool = True,
    max_pagination_pages: int = 5,
    top_k: Optional[int] = None,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    domain_rate_limits: Optional[Dict[str, float]] = None,
    proxies: Optional[List[str]] = None,
    cache=None,
    respect_robots: bool = False,
    progress=None,
    min_text_length: int = 0,
    similarity_threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    use_mmap: bool = False,
    mmap_max_mb: int = 256,
) -> List[dict]:
    """Async version of deep_search()."""
    pages = await deep_crawl(
        urls=urls, depth=depth, max_pages=max_pages,
        max_pages_per_domain=max_pages_per_domain,
        follow_external=follow_external, paginate=paginate,
        max_pagination_pages=max_pagination_pages, timeout=timeout,
        max_connections=max_connections, retries=retries, backoff=backoff,
        rate_limit=rate_limit, domain_rate_limits=domain_rate_limits,
        proxies=proxies, cache=cache,
        respect_robots=respect_robots, progress=progress,
        min_text_length=min_text_length, similarity_threshold=similarity_threshold,
        use_mmap=use_mmap, mmap_max_mb=mmap_max_mb,
    )
    return rank(pages, query, top_k=top_k, mode=mode,
                semantic_weight=semantic_weight, model_name=model_name)

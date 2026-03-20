"""
Crawler — async BFS depth crawler with cache, robots, proxy, and progress support.

Pure Python. Uses only:
  - asyncio           — concurrency
  - collections.deque — BFS queue
  - urllib.parse      — URL normalization
"""

import asyncio
import logging
from collections import deque
from typing import Dict, List, Optional
from urllib.parse import urldefrag, urlparse

from .deduplicator import Deduplicator
from .fetcher import fetch_one, _DEFAULT_HEADERS, _validate_urls
from .paginator import next_page_url
from .parser import parse
from .ratelimiter import DomainRateLimiter

import httpx

logger = logging.getLogger(__name__)


def _normalize(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    return parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl()


def _same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(base).netloc.lower()


async def deep_crawl(
    urls: List[str],
    depth: int = 2,
    max_pages: int = 100,
    max_pages_per_domain: int = 50,
    follow_external: bool = False,
    paginate: bool = True,
    max_pagination_pages: int = 5,
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
    use_mmap: bool = False,
    mmap_max_mb: int = 256,
) -> List[Dict]:
    """
    Async BFS depth crawler with cache, proxy rotation, robots.txt, and progress.

    Args:
        urls: Seed URLs to start crawling from.
        depth: How many link-levels deep to follow (0 = seed only).
        max_pages: Hard cap on total pages crawled across all domains.
        max_pages_per_domain: Cap per domain to avoid hammering one site.
        follow_external: If True, follow links to other domains.
        paginate: Auto-detect and follow pagination (next page links).
        max_pagination_pages: Max extra pages to follow per seed URL.
        timeout: Per-request timeout in seconds.
        max_connections: Max concurrent HTTP connections.
        retries: Retry attempts per URL.
        backoff: Exponential backoff multiplier.
        rate_limit: Global max requests per second (None = unlimited).
        domain_rate_limits: Per-domain RPS overrides e.g. {'example.com': 2.0}.
        proxies: List of proxy URLs for round-robin rotation.
        cache: TieredCache instance for response caching.
        respect_robots: Fetch and respect robots.txt per domain.
        progress: ProgressReporter instance for live updates.
        min_text_length: Skip pages with fewer extracted characters.
        similarity_threshold: Content similarity threshold for dedup (0-1).
        use_mmap: Store results in MMapStore (memory-mapped) instead of list.
        mmap_max_mb: Max MMapStore size in MB (default 256).

    Returns:
        List of parsed page dicts (url, title, text, links, meta, language).
    """
    from .robots import RobotsCache
    from .bridge import MMapStore

    _validate_urls(urls)

    robots = RobotsCache() if respect_robots else None
    dedup = Deduplicator(similarity_threshold=similarity_threshold)
    domain_limiter = DomainRateLimiter(
        default_rps=rate_limit,
        domain_rps=domain_rate_limits,
    )
    domain_counts: Dict[str, int] = {}
    results: List[Dict] = []
    store: Optional[MMapStore] = MMapStore(max_bytes=mmap_max_mb * 1024 * 1024) if use_mmap else None

    queue: deque = deque()
    for url in urls:
        norm = _normalize(url)
        queue.append((norm, 0, 1))
        dedup._seen_urls.add(norm)

    if progress:
        progress.start(len(urls))

    proxy_list = proxies or []
    proxy_index = 0
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections // 2,
    )

    def _next_proxy() -> Optional[str]:
        nonlocal proxy_index
        if not proxy_list:
            return None
        p = proxy_list[proxy_index % len(proxy_list)]
        proxy_index += 1
        return p

    def _make_client(proxy_url: Optional[str]) -> httpx.AsyncClient:
        kwargs = dict(limits=limits, headers=_DEFAULT_HEADERS,
                      http2=True, follow_redirects=True, verify=False)
        if proxy_url:
            kwargs["proxy"] = proxy_url
        return httpx.AsyncClient(**kwargs)

    async with _make_client(_next_proxy() if proxy_list else None) as client:
        while queue and len(results) < max_pages:
            batch = []
            while queue:
                batch.append(queue.popleft())

            fetch_tasks = [
                fetch_one(client, url, timeout, retries, backoff,
                          cache=cache, robots=robots, progress=progress)
                for url, _, _ in batch
            ]
            raw_pages = await asyncio.gather(*fetch_tasks)

            for (url, current_depth, page_num), raw in zip(batch, raw_pages):
                if len(results) >= max_pages:
                    break

                if raw.get("error") or not raw.get("html"):
                    logger.debug("Skipping %s: %s", url, raw.get("error"))
                    continue

                parsed = parse(raw["html"], raw["url"], min_text_length=min_text_length)
                if not parsed["text"]:
                    continue

                parsed["depth"] = current_depth
                parsed["page_num"] = page_num

                if dedup._is_content_duplicate(parsed):
                    logger.debug("Duplicate content skipped: %s", url)
                    continue

                dedup.register(parsed)
                domain = urlparse(url).netloc.lower()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                results.append(parsed)

                if store:
                    try:
                        store.write(parsed["url"], parsed["text"][:4096])
                    except OverflowError:
                        logger.warning("MMapStore full, continuing without mmap.")

                logger.info("[depth=%d page=%d] crawled: %s (%d total)",
                            current_depth, page_num, url, len(results))

                await domain_limiter.wait(url)

                # ── Enqueue child links ───────────────────────────────────────
                if current_depth < depth:
                    for link in parsed.get("links", []):
                        norm_link = _normalize(link)
                        if norm_link in dedup.seen_urls():
                            continue
                        if not follow_external and not _same_domain(norm_link, url):
                            continue
                        link_domain = urlparse(norm_link).netloc.lower()
                        if domain_counts.get(link_domain, 0) >= max_pages_per_domain:
                            continue
                        dedup._seen_urls.add(norm_link)
                        queue.append((norm_link, current_depth + 1, 1))
                        if progress:
                            progress.add_queued(1)

                # ── Enqueue pagination ────────────────────────────────────────
                if paginate and page_num < max_pagination_pages:
                    nxt = next_page_url(url, parsed.get("links", []), page_num)
                    if nxt:
                        norm_nxt = _normalize(nxt)
                        if norm_nxt not in dedup.seen_urls():
                            dedup._seen_urls.add(norm_nxt)
                            queue.append((norm_nxt, current_depth, page_num + 1))
                            if progress:
                                progress.add_queued(1)

    if progress:
        progress.done()

    if store:
        store.close()

    logger.info("Deep crawl complete. %d pages collected.", len(results))
    return results


def deep_crawl_sync(
    urls: List[str],
    depth: int = 2,
    max_pages: int = 100,
    max_pages_per_domain: int = 50,
    follow_external: bool = False,
    paginate: bool = True,
    max_pagination_pages: int = 5,
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
    use_mmap: bool = False,
    mmap_max_mb: int = 256,
) -> List[Dict]:
    """Sync wrapper for deep_crawl(). Safe in both sync and async contexts."""
    coro = deep_crawl(
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
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

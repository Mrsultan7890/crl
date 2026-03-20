"""
Fetcher — async HTTP fetching with cache, proxy rotation, robots.txt, and progress.

Pure Python + httpx only. No other language dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import httpx

from .cache import TieredCache
from .progress import ProgressReporter
from .robots import RobotsCache

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_SKIP_CONTENT_TYPES = {
    "application/pdf", "application/zip", "application/octet-stream",
    "application/x-tar", "application/x-gzip", "application/x-bzip2",
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml",
    "audio/mpeg", "audio/ogg", "video/mp4", "video/webm",
    "font/woff", "font/woff2", "application/font-woff",
}
_DEFAULT_HEADERS = {
    "User-Agent": "CRL-Crawler/1.0 (+https://github.com/crl-py/crl)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
}


async def fetch_one(
    client: httpx.AsyncClient,
    url: str,
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.5,
    cache: Optional[TieredCache] = None,
    robots: Optional[RobotsCache] = None,
    progress: Optional[ProgressReporter] = None,
) -> Dict:
    # ── Cache check ───────────────────────────────────────────────────────────
    if cache:
        hit = cache.get(url)
        if hit:
            logger.debug("Cache hit: %s", url)
            if progress:
                progress.cached(url)
            return hit

    # ── robots.txt check ──────────────────────────────────────────────────────
    if robots:
        allowed = await robots.is_allowed(url, client)
        if not allowed:
            logger.info("Blocked by robots.txt: %s", url)
            if progress:
                progress.skipped(url, reason="robots.txt")
            return {"url": url, "original_url": url, "status": None,
                    "html": None, "headers": {}, "error": "blocked by robots.txt"}

        # Respect Crawl-delay
        delay = robots.crawl_delay(url)
        if delay > 0:
            await asyncio.sleep(delay)

    # ── Fetch with retry ──────────────────────────────────────────────────────
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, timeout=timeout, follow_redirects=True)
            if response.status_code in _RETRY_STATUSES and attempt < retries:
                wait = backoff ** attempt
                logger.warning("HTTP %s for %s — retrying in %.1fs (%d/%d)",
                               response.status_code, url, wait, attempt, retries)
                await asyncio.sleep(wait)
                continue

            # Content-type filter — skip binary/media responses
            ct = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct and ct in _SKIP_CONTENT_TYPES:
                logger.debug("Skipping non-HTML content-type '%s': %s", ct, url)
                if progress:
                    progress.skipped(url, reason=f"content-type:{ct}")
                return {"url": url, "original_url": url, "status": response.status_code,
                        "html": None, "headers": dict(response.headers),
                        "error": f"skipped content-type: {ct}"}

            result = {
                "url": str(response.url),
                "original_url": url,
                "status": response.status_code,
                "html": response.text,
                "headers": dict(response.headers),
                "error": None,
            }
            if cache:
                cache.set(url, result)
            if progress:
                progress.fetched(url)
            return result

        except httpx.TimeoutException:
            last_error = f"Timeout after {timeout}s"
        except httpx.TooManyRedirects:
            last_error = "Too many redirects"
            break
        except httpx.InvalidURL:
            last_error = f"Invalid URL: {url}"
            break
        except httpx.RequestError as exc:
            last_error = str(exc)

        if attempt < retries:
            wait = backoff ** attempt
            logger.warning("Error fetching %s: %s — retrying in %.1fs (%d/%d)",
                           url, last_error, wait, attempt, retries)
            await asyncio.sleep(wait)

    logger.error("Failed to fetch %s after %d attempts: %s", url, retries, last_error)
    if progress:
        progress.error(url, message=last_error or "")
    return {"url": url, "original_url": url, "status": None,
            "html": None, "headers": {}, "error": last_error}


async def fetch_all(
    urls: List[str],
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache: Optional[TieredCache] = None,
    respect_robots: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict]:
    """
    Fetch all URLs concurrently with optional cache, proxy rotation,
    robots.txt compliance, rate limiting, and progress reporting.

    Args:
        urls: List of URLs to fetch.
        timeout: Per-request timeout in seconds.
        max_connections: Max concurrent HTTP connections.
        retries: Retry attempts per URL on failure.
        backoff: Exponential backoff multiplier.
        rate_limit: Max requests per second (None = unlimited).
        proxies: List of proxy URLs to rotate (e.g. ['http://proxy1:8080']).
                 Rotated round-robin across requests.
        cache: TieredCache instance for caching responses.
        respect_robots: If True, fetch and respect robots.txt per domain.
        progress: ProgressReporter instance for live progress updates.
    """
    _validate_urls(urls)

    robots = RobotsCache() if respect_robots else None
    if progress:
        progress.start(len(urls))

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections // 2,
    )

    proxy_list = proxies or []
    results = []

    async def _make_client(proxy_url: Optional[str]) -> httpx.AsyncClient:
        kwargs = dict(limits=limits, headers=_DEFAULT_HEADERS,
                      http2=True, follow_redirects=True, verify=False)
        if proxy_url:
            kwargs["proxy"] = proxy_url
        return httpx.AsyncClient(**kwargs)

    if proxy_list:
        # Proxy rotation: each URL gets a proxy round-robin
        results = await _fetch_with_proxy_rotation(
            urls, proxy_list, timeout, retries, backoff,
            rate_limit, cache, robots, progress, limits,
        )
    else:
        async with await _make_client(None) as client:
            if rate_limit:
                results = await _fetch_rate_limited(
                    client, urls, timeout, retries, backoff,
                    rate_limit, cache, robots, progress,
                )
            else:
                tasks = [
                    fetch_one(client, url, timeout, retries, backoff,
                              cache=cache, robots=robots, progress=progress)
                    for url in urls
                ]
                results = await asyncio.gather(*tasks)

    if progress:
        progress.done()
    return results


async def _fetch_with_proxy_rotation(
    urls: List[str],
    proxies: List[str],
    timeout: int,
    retries: int,
    backoff: float,
    rate_limit: Optional[float],
    cache: Optional[TieredCache],
    robots: Optional[RobotsCache],
    progress: Optional[ProgressReporter],
    limits: httpx.Limits,
) -> List[Dict]:
    """Round-robin proxy rotation — each URL gets the next proxy in the list."""
    interval = (1.0 / rate_limit) if rate_limit else 0.0
    results = []
    for i, url in enumerate(urls):
        proxy_url = proxies[i % len(proxies)]
        async with httpx.AsyncClient(
            limits=limits, headers=_DEFAULT_HEADERS,
            http2=True, follow_redirects=True, verify=False, proxy=proxy_url,
        ) as client:
            start = time.monotonic()
            result = await fetch_one(client, url, timeout, retries, backoff,
                                     cache=cache, robots=robots, progress=progress)
            results.append(result)
            if interval > 0:
                elapsed = time.monotonic() - start
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
    return results


async def _fetch_rate_limited(
    client: httpx.AsyncClient,
    urls: List[str],
    timeout: int,
    retries: int,
    backoff: float,
    rate_limit: float,
    cache: Optional[TieredCache],
    robots: Optional[RobotsCache],
    progress: Optional[ProgressReporter],
) -> List[Dict]:
    interval = 1.0 / rate_limit
    results = []
    for url in urls:
        start = time.monotonic()
        result = await fetch_one(client, url, timeout, retries, backoff,
                                 cache=cache, robots=robots, progress=progress)
        results.append(result)
        elapsed = time.monotonic() - start
        sleep_for = interval - elapsed
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
    return results


def fetch(
    urls: List[str],
    timeout: int = 10,
    max_connections: int = 20,
    retries: int = 3,
    backoff: float = 1.5,
    rate_limit: Optional[float] = None,
    proxies: Optional[List[str]] = None,
    cache: Optional[TieredCache] = None,
    respect_robots: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict]:
    """
    Sync entry point. Safe to call from both sync and async contexts.
    """
    coro = fetch_all(
        urls, timeout=timeout, max_connections=max_connections,
        retries=retries, backoff=backoff, rate_limit=rate_limit,
        proxies=proxies, cache=cache, respect_robots=respect_robots,
        progress=progress,
    )
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def _validate_urls(urls: List[str]) -> None:
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme for '{url}'. Only http/https allowed.")

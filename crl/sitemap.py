"""
Sitemap parser — fetch and parse sitemap.xml / sitemap index files.

Supports:
  - /sitemap.xml auto-discovery
  - <sitemapindex> (nested sitemaps)
  - <urlset> (leaf URL lists)
  - sitemap: entries in robots.txt
  - gzip compressed sitemaps
"""

import asyncio
import gzip
import logging
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import httpx

from .fetcher import _DEFAULT_HEADERS

logger = logging.getLogger(__name__)

_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}

_SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/sitemap.xml"]


def _base(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _parse_urlset(xml: str) -> List[str]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    urls = []
    for loc in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        if loc.text:
            urls.append(loc.text.strip())
    return urls


def _parse_sitemapindex(xml: str) -> List[str]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    locs = []
    for loc in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        if loc.text:
            locs.append(loc.text.strip())
    return locs


def _is_index(xml: str) -> bool:
    return "<sitemapindex" in xml


async def _fetch_xml(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, timeout=10, follow_redirects=True)
        if r.status_code != 200:
            return None
        content = r.content
        # Handle gzip
        if url.endswith(".gz") or r.headers.get("content-encoding") == "gzip":
            try:
                content = gzip.decompress(content)
            except Exception:
                pass
        return content.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.debug("Sitemap fetch failed %s: %s", url, exc)
        return None


async def fetch_sitemap_urls(
    base_url: str,
    max_urls: int = 10000,
    timeout: int = 10,
) -> List[str]:
    """
    Fetch all URLs from a site's sitemap(s).

    Tries /sitemap.xml, /sitemap_index.xml, and robots.txt Sitemap: entries.
    Recursively follows sitemap index files.
    Follows redirects to resolve the canonical base URL.

    Args:
        base_url: Any URL on the target site (only scheme+host used).
        max_urls: Max URLs to return.
        timeout: HTTP timeout per request.

    Returns:
        Deduplicated list of URLs found in sitemaps.
    """
    async with httpx.AsyncClient(
        headers=_DEFAULT_HEADERS, verify=False,
        follow_redirects=True, timeout=timeout,
    ) as client:
        # Resolve canonical base by following redirects on the root URL
        canonical_base = await _resolve_base(client, base_url)

        # 1. Try to find sitemap URLs from robots.txt
        sitemap_urls = await _discover_from_robots(client, canonical_base)

        # 2. Fall back to common paths
        if not sitemap_urls:
            sitemap_urls = [canonical_base + p for p in _SITEMAP_PATHS]

        all_urls: List[str] = []
        visited_sitemaps = set()

        async def _process(sm_url: str) -> None:
            if sm_url in visited_sitemaps or len(all_urls) >= max_urls:
                return
            visited_sitemaps.add(sm_url)
            xml = await _fetch_xml(client, sm_url)
            if not xml:
                return
            if _is_index(xml):
                child_sitemaps = _parse_sitemapindex(xml)
                tasks = [_process(u) for u in child_sitemaps]
                await asyncio.gather(*tasks)
            else:
                urls = _parse_urlset(xml)
                all_urls.extend(urls[:max_urls - len(all_urls)])

        await asyncio.gather(*[_process(u) for u in sitemap_urls])

    seen = set()
    result = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result[:max_urls]


async def _resolve_base(client: httpx.AsyncClient, url: str) -> str:
    """Follow redirects on root URL to get canonical base (scheme+host)."""
    try:
        r = await client.get(_base(url) + "/", timeout=5)
        return _base(str(r.url))
    except Exception:
        return _base(url)


async def _discover_from_robots(client: httpx.AsyncClient, base: str) -> List[str]:
    """Extract Sitemap: lines from robots.txt."""
    try:
        r = await client.get(f"{base}/robots.txt", timeout=5)
        if r.status_code != 200:
            return []
        urls = []
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                url = line.split(":", 1)[1].strip()
                if url:
                    urls.append(url)
        return urls
    except Exception:
        return []


def fetch_sitemap_urls_sync(
    base_url: str,
    max_urls: int = 10000,
    timeout: int = 10,
) -> List[str]:
    """Sync wrapper for fetch_sitemap_urls()."""
    coro = fetch_sitemap_urls(base_url, max_urls=max_urls, timeout=timeout)
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

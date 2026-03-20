"""
Robots — pure Python robots.txt fetching, parsing and caching.

Uses only stdlib:
  - urllib.robotparser — robots.txt parsing
  - urllib.parse       — URL handling
  - asyncio            — async fetch
  - functools          — per-domain singleton cache
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = "CRL-Crawler"
_ROBOTS_TTL = 3600  # re-fetch robots.txt after 1 hour


class RobotsCache:
    """
    Async-safe per-domain robots.txt cache.
    Fetches and parses robots.txt once per domain, respects TTL.
    """

    def __init__(self, ttl: int = _ROBOTS_TTL, user_agent: str = _USER_AGENT):
        self._ttl = ttl
        self._user_agent = user_agent
        # domain → (RobotFileParser, fetched_at)
        self._cache: Dict[str, tuple] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        async with self._global_lock:
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
            return self._locks[domain]

    async def is_allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        """
        Return True if the URL is allowed to be crawled per robots.txt.
        Always returns True on fetch errors (fail open).
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"

        lock = await self._get_lock(domain)
        async with lock:
            cached = self._cache.get(domain)
            if cached:
                parser, fetched_at = cached
                if (time.monotonic() - fetched_at) < self._ttl:
                    return parser.can_fetch(self._user_agent, url)

            parser = await self._fetch_robots(robots_url, client)
            self._cache[domain] = (parser, time.monotonic())

        return parser.can_fetch(self._user_agent, url)

    async def _fetch_robots(self, robots_url: str, client: httpx.AsyncClient) -> RobotFileParser:
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = await client.get(robots_url, timeout=5, follow_redirects=True)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                logger.debug("Fetched robots.txt from %s", robots_url)
            elif response.status_code == 404:
                # No robots.txt — everything allowed
                parser.parse([])
            else:
                logger.debug("robots.txt returned %s for %s — allowing all",
                             response.status_code, robots_url)
                parser.parse([])
        except Exception as exc:
            logger.debug("Could not fetch robots.txt from %s: %s — allowing all", robots_url, exc)
            parser.parse([])
        return parser

    def crawl_delay(self, url: str) -> float:
        """Return Crawl-delay for domain if specified, else 0.0."""
        domain = urlparse(url).netloc.lower()
        cached = self._cache.get(domain)
        if not cached:
            return 0.0
        parser, _ = cached
        delay = parser.crawl_delay(self._user_agent)
        return float(delay) if delay is not None else 0.0

    def clear(self) -> None:
        self._cache.clear()

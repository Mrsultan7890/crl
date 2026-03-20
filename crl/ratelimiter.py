"""
Per-domain rate limiter — asyncio-based token bucket per domain.

Prevents hammering a single domain while allowing full speed on others.
Thread-safe via asyncio.Lock per domain.
"""

import asyncio
import time
from typing import Dict, Optional
from urllib.parse import urlparse


class DomainRateLimiter:
    """
    Per-domain async rate limiter using token bucket algorithm.

    Each domain gets its own bucket. Domains not in the limit map
    use the default rate (or unlimited if default is None).

    Args:
        default_rps: Default max requests/sec per domain (None = unlimited).
        domain_rps: Per-domain overrides e.g. {'example.com': 2.0}.

    Example:
        limiter = DomainRateLimiter(default_rps=5.0, domain_rps={'slow.com': 1.0})
        async with limiter.acquire('https://slow.com/page'):
            ...  # max 1 req/sec for slow.com
    """

    def __init__(
        self,
        default_rps: Optional[float] = None,
        domain_rps: Optional[Dict[str, float]] = None,
    ):
        self._default_rps = default_rps
        self._domain_rps = domain_rps or {}
        self._last_request: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    def _interval(self, domain: str) -> float:
        rps = self._domain_rps.get(domain, self._default_rps)
        return (1.0 / rps) if rps else 0.0

    async def wait(self, url: str) -> None:
        """
        Wait if needed to respect rate limit for this URL's domain.
        Call before each request.
        """
        domain = urlparse(url).netloc.lower()
        interval = self._interval(domain)
        if interval <= 0:
            return

        lock = self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            wait_for = interval - (now - last)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request[domain] = time.monotonic()

    def set_domain_rps(self, domain: str, rps: float) -> None:
        """Set or update rate limit for a specific domain."""
        self._domain_rps[domain] = rps

    def reset(self, domain: Optional[str] = None) -> None:
        """Reset last-request timestamps. Pass domain to reset one, None for all."""
        if domain:
            self._last_request.pop(domain, None)
        else:
            self._last_request.clear()

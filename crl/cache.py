"""
Cache — pure Python disk + memory caching for crawled pages.

Uses only stdlib:
  - shelve    — persistent disk cache (backed by dbm)
  - hashlib   — cache key generation
  - time      — TTL expiry
  - threading — thread-safe in-memory LRU cache
"""

import hashlib
import logging
import shelve
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v1"


def _cache_key(url: str) -> str:
    return f"{_CACHE_VERSION}:{hashlib.sha256(url.encode()).hexdigest()}"


class MemoryCache:
    """
    Thread-safe in-memory LRU cache with TTL support.

    Args:
        max_size: Max number of entries to keep in memory.
        ttl: Time-to-live in seconds. 0 = never expire.
    """

    def __init__(self, max_size: int = 512, ttl: int = 3600):
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, url: str) -> Optional[Dict]:
        key = _cache_key(url)
        with self._lock:
            if key not in self._store:
                return None
            entry, ts = self._store[key]
            if self._ttl and (time.monotonic() - ts) > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry

    def set(self, url: str, page: Dict) -> None:
        key = _cache_key(url)
        with self._lock:
            self._store[key] = (page, time.monotonic())
            self._store.move_to_end(key)
            if len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("LRU evicted cache entry: %s", evicted_key)

    def invalidate(self, url: str) -> None:
        key = _cache_key(url)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class DiskCache:
    """
    Persistent disk cache using stdlib shelve (dbm-backed).
    Thread-safe via lock. Supports TTL expiry.

    Args:
        path: File path prefix for the shelve database (e.g. '.crl_cache').
        ttl: Time-to-live in seconds. 0 = never expire.
    """

    def __init__(self, path: str = ".crl_cache", ttl: int = 86400):
        self._path = path
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, url: str) -> Optional[Dict]:
        key = _cache_key(url)
        with self._lock:
            try:
                with shelve.open(self._path, flag="r") as db:
                    if key not in db:
                        return None
                    entry, ts = db[key]
                    if self._ttl and (time.time() - ts) > self._ttl:
                        return None
                    return entry
            except Exception:
                return None

    def set(self, url: str, page: Dict) -> None:
        key = _cache_key(url)
        with self._lock:
            try:
                with shelve.open(self._path) as db:
                    db[key] = (page, time.time())
            except Exception as exc:
                logger.warning("DiskCache write failed for %s: %s", url, exc)

    def invalidate(self, url: str) -> None:
        key = _cache_key(url)
        with self._lock:
            try:
                with shelve.open(self._path) as db:
                    db.pop(key, None)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            try:
                with shelve.open(self._path) as db:
                    db.clear()
            except Exception:
                pass


class TieredCache:
    """
    Two-level cache: memory (L1) → disk (L2).
    Read hits in L1 are fast. Misses fall through to L2 and promote to L1.

    Args:
        memory_size: Max entries in memory cache.
        memory_ttl: Memory cache TTL in seconds.
        disk_path: Disk cache file path prefix.
        disk_ttl: Disk cache TTL in seconds.
    """

    def __init__(
        self,
        memory_size: int = 512,
        memory_ttl: int = 3600,
        disk_path: str = ".crl_cache",
        disk_ttl: int = 86400,
    ):
        self._mem = MemoryCache(max_size=memory_size, ttl=memory_ttl)
        self._disk = DiskCache(path=disk_path, ttl=disk_ttl)

    def get(self, url: str) -> Optional[Dict]:
        hit = self._mem.get(url)
        if hit:
            return hit
        hit = self._disk.get(url)
        if hit:
            self._mem.set(url, hit)  # promote to L1
        return hit

    def set(self, url: str, page: Dict) -> None:
        self._mem.set(url, page)
        self._disk.set(url, page)

    def invalidate(self, url: str) -> None:
        self._mem.invalidate(url)
        self._disk.invalidate(url)

    def clear(self) -> None:
        self._mem.clear()
        self._disk.clear()

"""
Deduplicator — pure Python content deduplication using hashlib + shingle fingerprinting.

Two levels of dedup:
  1. URL-level  — exact same URL never crawled twice
  2. Content-level — near-duplicate pages filtered via shingle hashing (no external deps)
"""

import hashlib
import re
from typing import Dict, List, Set

_WHITESPACE = re.compile(r"\s+")


def _fingerprint(text: str, shingle_size: int = 8) -> Set[int]:
    """
    Build a set of shingle hashes from text.
    Each shingle is a sliding window of `shingle_size` words hashed with xxhash-like
    approach using only stdlib hashlib (sha1 truncated to 64-bit int).
    """
    words = _WHITESPACE.sub(" ", text.lower()).split()
    if len(words) < shingle_size:
        h = int(hashlib.sha1(" ".join(words).encode()).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF
        return {h}
    return {
        int(hashlib.sha1(" ".join(words[i: i + shingle_size]).encode()).hexdigest(), 16)
        & 0xFFFFFFFFFFFFFFFF
        for i in range(len(words) - shingle_size + 1)
    }


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class Deduplicator:
    """
    Stateful deduplicator. Tracks seen URLs and content fingerprints across
    multiple crawl batches.

    Args:
        similarity_threshold: Pages with Jaccard similarity above this are
                              considered duplicates (default 0.85).
        shingle_size: Number of words per shingle (default 8).
    """

    def __init__(self, similarity_threshold: float = 0.85, shingle_size: int = 8):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        self._threshold = similarity_threshold
        self._shingle_size = shingle_size
        self._seen_urls: Set[str] = set()
        self._fingerprints: List[Set[int]] = []

    def is_duplicate(self, page: Dict) -> bool:
        """Return True if page is a URL or content duplicate."""
        url = page.get("url", "")
        if url in self._seen_urls:
            return True
        return self._is_content_duplicate(page)

    def _is_content_duplicate(self, page: Dict) -> bool:
        """Return True if page content matches a previously seen page."""
        text = page.get("text", "")
        if not text:
            return False
        fp = _fingerprint(text, self._shingle_size)
        for seen_fp in self._fingerprints:
            if _jaccard(fp, seen_fp) >= self._threshold:
                return True
        return False

    def register(self, page: Dict) -> None:
        """Mark a page as seen so future duplicates are detected."""
        url = page.get("url", "")
        if url:
            self._seen_urls.add(url)
        text = page.get("text", "")
        if text:
            self._fingerprints.append(_fingerprint(text, self._shingle_size))

    def filter(self, pages: List[Dict]) -> List[Dict]:
        """
        Filter a list of pages, removing duplicates.
        Registers each kept page automatically.
        """
        unique = []
        for page in pages:
            if not self.is_duplicate(page):
                self.register(page)
                unique.append(page)
        return unique

    def seen_urls(self) -> Set[str]:
        return frozenset(self._seen_urls)

    def reset(self) -> None:
        self._seen_urls.clear()
        self._fingerprints.clear()

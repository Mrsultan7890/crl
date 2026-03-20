"""
Progress — pure Python streaming progress reporter.

Uses only stdlib:
  - sys       — stderr output
  - time      — ETA calculation
  - threading — thread-safe counters
  - dataclasses — clean event structure
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CrawlEvent:
    """Emitted for every crawl action."""
    event: str          # "start" | "fetched" | "cached" | "skipped" | "error" | "done"
    url: str = ""
    total: int = 0      # total URLs queued so far
    done: int = 0       # completed (success + error)
    errors: int = 0
    cached: int = 0
    elapsed: float = 0.0
    eta: Optional[float] = None   # seconds remaining
    message: str = ""


ProgressCallback = Callable[[CrawlEvent], None]


class ProgressReporter:
    """
    Thread-safe progress tracker. Emits CrawlEvent to a callback on every update.
    Built-in console reporter included — pass reporter.console_callback as callback.

    Args:
        total: Expected total URLs (can be updated dynamically).
        callback: Called with CrawlEvent on every update.
        stream: Output stream for console reporter (default stderr).
    """

    def __init__(
        self,
        total: int = 0,
        callback: Optional[ProgressCallback] = None,
        stream=None,
    ):
        self._total = total
        self._done = 0
        self._errors = 0
        self._cached = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._callback = callback or self._console_callback
        self._stream = stream or sys.stderr

    def start(self, total: int) -> None:
        with self._lock:
            self._total = total
            self._start = time.monotonic()
        self._emit("start", total=total)

    def fetched(self, url: str) -> None:
        with self._lock:
            self._done += 1
        self._emit("fetched", url=url)

    def cached(self, url: str) -> None:
        with self._lock:
            self._done += 1
            self._cached += 1
        self._emit("cached", url=url)

    def skipped(self, url: str, reason: str = "") -> None:
        with self._lock:
            self._done += 1
        self._emit("skipped", url=url, message=reason)

    def error(self, url: str, message: str = "") -> None:
        with self._lock:
            self._done += 1
            self._errors += 1
        self._emit("error", url=url, message=message)

    def add_queued(self, count: int = 1) -> None:
        """Dynamically increase total when new links are discovered."""
        with self._lock:
            self._total += count

    def done(self) -> None:
        self._emit("done")

    def _emit(self, event: str, url: str = "", message: str = "", total: int = 0) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._start
            done = self._done
            ttl = total or self._total
            errors = self._errors
            cached = self._cached

        eta = None
        if done > 0 and ttl > done:
            rate = done / elapsed if elapsed > 0 else 0
            eta = (ttl - done) / rate if rate > 0 else None

        ev = CrawlEvent(
            event=event, url=url, total=ttl, done=done,
            errors=errors, cached=cached, elapsed=elapsed,
            eta=eta, message=message,
        )
        try:
            self._callback(ev)
        except Exception:
            pass

    def _console_callback(self, ev: CrawlEvent) -> None:
        """Built-in human-readable console progress output to stderr."""
        stream = self._stream
        if ev.event == "start":
            print(f"\n[CRL] Starting crawl — {ev.total} URLs queued", file=stream)
        elif ev.event == "done":
            print(
                f"\n[CRL] Done — {ev.done} crawled, {ev.cached} cached, "
                f"{ev.errors} errors in {ev.elapsed:.1f}s",
                file=stream,
            )
        elif ev.event == "error":
            print(f"\r[CRL] ✗ {ev.url[:60]} — {ev.message}", file=stream)
        else:
            eta_str = f" ETA {ev.eta:.0f}s" if ev.eta else ""
            bar = _progress_bar(ev.done, ev.total)
            print(
                f"\r[CRL] {bar} {ev.done}/{ev.total}{eta_str} | "
                f"cached={ev.cached} err={ev.errors} | {ev.url[:50]}",
                end="",
                file=stream,
                flush=True,
            )


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return f"[{'?' * width}]"
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}]"

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crl.cache import MemoryCache, DiskCache, TieredCache, _cache_key
from crl.progress import ProgressReporter, CrawlEvent, _progress_bar
from crl.robots import RobotsCache


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_cache_key_is_deterministic():
    assert _cache_key("https://example.com") == _cache_key("https://example.com")


def test_cache_key_differs_for_different_urls():
    assert _cache_key("https://a.com") != _cache_key("https://b.com")


def test_memory_cache_set_get():
    c = MemoryCache()
    page = {"url": "https://a.com", "text": "hello"}
    c.set("https://a.com", page)
    assert c.get("https://a.com") == page


def test_memory_cache_miss_returns_none():
    c = MemoryCache()
    assert c.get("https://notcached.com") is None


def test_memory_cache_ttl_expiry():
    c = MemoryCache(ttl=1)
    c.set("https://a.com", {"text": "hi"})
    assert c.get("https://a.com") is not None
    # Manually expire by manipulating internal store
    key = _cache_key("https://a.com")
    entry, _ = c._store[key]
    c._store[key] = (entry, time.monotonic() - 2)
    assert c.get("https://a.com") is None


def test_memory_cache_lru_eviction():
    c = MemoryCache(max_size=2)
    c.set("https://a.com", {"text": "a"})
    c.set("https://b.com", {"text": "b"})
    c.set("https://c.com", {"text": "c"})  # evicts a
    assert c.get("https://a.com") is None
    assert c.get("https://b.com") is not None
    assert c.get("https://c.com") is not None


def test_memory_cache_invalidate():
    c = MemoryCache()
    c.set("https://a.com", {"text": "hi"})
    c.invalidate("https://a.com")
    assert c.get("https://a.com") is None


def test_memory_cache_clear():
    c = MemoryCache()
    c.set("https://a.com", {"text": "a"})
    c.set("https://b.com", {"text": "b"})
    c.clear()
    assert len(c) == 0


def test_memory_cache_len():
    c = MemoryCache()
    c.set("https://a.com", {"text": "a"})
    c.set("https://b.com", {"text": "b"})
    assert len(c) == 2


def test_disk_cache_set_get(tmp_path):
    c = DiskCache(path=str(tmp_path / "cache"))
    page = {"url": "https://a.com", "text": "hello disk"}
    c.set("https://a.com", page)
    assert c.get("https://a.com") == page


def test_disk_cache_miss(tmp_path):
    c = DiskCache(path=str(tmp_path / "cache"))
    assert c.get("https://notcached.com") is None


def test_disk_cache_ttl_expiry(tmp_path):
    c = DiskCache(path=str(tmp_path / "cache"), ttl=1)
    c.set("https://a.com", {"text": "hi"})
    # Manually write expired entry
    import shelve, time as t
    key = _cache_key("https://a.com")
    with shelve.open(str(tmp_path / "cache")) as db:
        db[key] = ({"text": "hi"}, t.time() - 10)
    assert c.get("https://a.com") is None


def test_disk_cache_invalidate(tmp_path):
    c = DiskCache(path=str(tmp_path / "cache"))
    c.set("https://a.com", {"text": "hi"})
    c.invalidate("https://a.com")
    assert c.get("https://a.com") is None


def test_disk_cache_clear(tmp_path):
    c = DiskCache(path=str(tmp_path / "cache"))
    c.set("https://a.com", {"text": "a"})
    c.clear()
    assert c.get("https://a.com") is None


def test_tiered_cache_l1_hit(tmp_path):
    c = TieredCache(disk_path=str(tmp_path / "cache"))
    page = {"url": "https://a.com", "text": "tiered"}
    c.set("https://a.com", page)
    # L1 hit — disk not needed
    assert c._mem.get("https://a.com") == page
    assert c.get("https://a.com") == page


def test_tiered_cache_l2_promotion(tmp_path):
    c = TieredCache(disk_path=str(tmp_path / "cache"))
    page = {"url": "https://a.com", "text": "tiered"}
    c._disk.set("https://a.com", page)
    # L1 miss → L2 hit → promotes to L1
    result = c.get("https://a.com")
    assert result == page
    assert c._mem.get("https://a.com") == page


def test_tiered_cache_clear(tmp_path):
    c = TieredCache(disk_path=str(tmp_path / "cache"))
    c.set("https://a.com", {"text": "a"})
    c.clear()
    assert c.get("https://a.com") is None


# ── Progress ──────────────────────────────────────────────────────────────────

def test_progress_bar_full():
    assert "█" * 20 in _progress_bar(20, 20)


def test_progress_bar_empty():
    assert "░" * 20 in _progress_bar(0, 20)


def test_progress_bar_zero_total():
    assert "?" in _progress_bar(0, 0)


def test_progress_reporter_emits_events():
    events = []
    p = ProgressReporter(callback=events.append)
    p.start(10)
    p.fetched("https://a.com")
    p.cached("https://b.com")
    p.error("https://c.com", "timeout")
    p.skipped("https://d.com", "robots.txt")
    p.done()

    event_types = [e.event for e in events]
    assert "start" in event_types
    assert "fetched" in event_types
    assert "cached" in event_types
    assert "error" in event_types
    assert "skipped" in event_types
    assert "done" in event_types


def test_progress_reporter_counts():
    events = []
    p = ProgressReporter(callback=events.append)
    p.start(5)
    p.fetched("https://a.com")
    p.fetched("https://b.com")
    p.error("https://c.com", "err")
    p.cached("https://d.com")

    done_events = [e for e in events if e.event in ("fetched", "error", "cached")]
    assert done_events[-1].done == 4
    assert done_events[-1].errors == 1
    assert done_events[-1].cached == 1


def test_progress_add_queued():
    events = []
    p = ProgressReporter(total=5, callback=events.append)
    p.add_queued(3)
    p.fetched("https://a.com")
    fetched_event = [e for e in events if e.event == "fetched"][0]
    assert fetched_event.total == 8


def test_progress_eta_calculated():
    events = []
    p = ProgressReporter(total=10, callback=events.append)
    p._start = time.monotonic() - 5  # simulate 5s elapsed
    p._done = 5
    p.fetched("https://a.com")
    fetched = [e for e in events if e.event == "fetched"][0]
    assert fetched.eta is not None
    assert fetched.eta > 0


def test_progress_console_output(capsys):
    p = ProgressReporter()
    p.start(3)
    p.fetched("https://a.com")
    p.done()
    captured = capsys.readouterr()
    assert "CRL" in captured.err


# ── Robots ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_robots_allows_when_no_disallow():
    robots_txt = "User-agent: *\nAllow: /"
    mock_response = MagicMock(status_code=200, text=robots_txt)
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    rc = RobotsCache()
    allowed = await rc.is_allowed("https://example.com/page", mock_client)
    assert allowed is True


@pytest.mark.asyncio
async def test_robots_blocks_disallowed():
    robots_txt = "User-agent: *\nDisallow: /private/"
    mock_response = MagicMock(status_code=200, text=robots_txt)
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    rc = RobotsCache()
    allowed = await rc.is_allowed("https://example.com/private/secret", mock_client)
    assert allowed is False


@pytest.mark.asyncio
async def test_robots_allows_on_404():
    mock_response = MagicMock(status_code=404, text="")
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    rc = RobotsCache()
    allowed = await rc.is_allowed("https://example.com/anything", mock_client)
    assert allowed is True


@pytest.mark.asyncio
async def test_robots_allows_on_fetch_error():
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("network error")

    rc = RobotsCache()
    allowed = await rc.is_allowed("https://example.com/page", mock_client)
    assert allowed is True


@pytest.mark.asyncio
async def test_robots_caches_per_domain():
    robots_txt = "User-agent: *\nAllow: /"
    mock_response = MagicMock(status_code=200, text=robots_txt)
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    rc = RobotsCache()
    await rc.is_allowed("https://example.com/page1", mock_client)
    await rc.is_allowed("https://example.com/page2", mock_client)
    # robots.txt should only be fetched once per domain
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_robots_crawl_delay():
    robots_txt = "User-agent: CRL-Crawler\nCrawl-delay: 2"
    mock_response = MagicMock(status_code=200, text=robots_txt)
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    rc = RobotsCache()
    await rc.is_allowed("https://example.com/page", mock_client)
    assert rc.crawl_delay("https://example.com/page") == 2.0


def test_robots_clear():
    rc = RobotsCache()
    rc._cache["example.com"] = (MagicMock(), time.monotonic())
    rc.clear()
    assert len(rc._cache) == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_help():
    from crl.cli import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_crawl_args():
    from crl.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "crawl", "https://example.com",
        "--query", "python async",
        "--top-k", "5",
        "--mode", "keyword",
        "--no-progress",
    ])
    assert args.command == "crawl"
    assert args.urls == ["https://example.com"]
    assert args.query == "python async"
    assert args.top_k == 5
    assert args.mode == "keyword"
    assert args.no_progress is True


def test_cli_deep_args():
    from crl.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "deep", "https://example.com",
        "--query", "python",
        "--depth", "3",
        "--max-pages", "50",
        "--no-progress",
    ])
    assert args.command == "deep"
    assert args.depth == 3
    assert args.max_pages == 50


def test_cli_proxy_args():
    from crl.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "crawl", "https://example.com",
        "--query", "test",
        "--proxy", "http://p1:8080",
        "--proxy", "http://p2:8080",
        "--no-progress",
    ])
    assert args.proxies == ["http://p1:8080", "http://p2:8080"]


def test_cli_main_crawl(tmp_path):
    from crl.cli import main
    mock_results = [{"url": "https://example.com", "title": "Test",
                     "text": "hello", "relevance_score": 0.9,
                     "keyword_score": 0.8, "semantic_score": 0.95, "language": "en"}]
    with patch("crl.crawl", return_value=mock_results):
        out_file = str(tmp_path / "out.json")
        code = main([
            "crawl", "https://example.com",
            "--query", "python",
            "--no-progress",
            "--out", out_file,
        ])
    assert code == 0
    data = json.loads((tmp_path / "out.json").read_text())
    assert data[0]["url"] == "https://example.com"


def test_cli_main_invalid_url():
    from crl.cli import main
    with patch("crl.crawl", side_effect=ValueError("Invalid URL scheme")):
        code = main(["crawl", "ftp://bad.com", "--query", "test", "--no-progress"])
    assert code == 1

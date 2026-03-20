import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crl.deduplicator import Deduplicator, _fingerprint, _jaccard
from crl.paginator import (
    next_page_url, build_page_urls, detect_pagination_links,
    _increment_query_param, _increment_path,
)
from crl.crawler import _normalize, _same_domain, deep_crawl


# ── Deduplicator ──────────────────────────────────────────────────────────────

def test_fingerprint_returns_set_of_ints():
    fp = _fingerprint("hello world python async crawl relevance layers test")
    assert isinstance(fp, set)
    assert all(isinstance(x, int) for x in fp)


def test_fingerprint_short_text():
    fp = _fingerprint("hi")
    assert len(fp) == 1


def test_jaccard_identical():
    a = {1, 2, 3}
    assert _jaccard(a, a) == pytest.approx(1.0)


def test_jaccard_disjoint():
    assert _jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_jaccard_partial():
    score = _jaccard({1, 2, 3}, {2, 3, 4})
    assert 0.0 < score < 1.0


def test_jaccard_empty():
    assert _jaccard(set(), {1, 2}) == pytest.approx(0.0)


def test_dedup_url_duplicate():
    d = Deduplicator()
    page = {"url": "https://a.com", "text": "some unique content here for testing purposes only"}
    d.register(page)
    assert d.is_duplicate({"url": "https://a.com", "text": "completely different text"})


def test_dedup_content_duplicate():
    d = Deduplicator(similarity_threshold=0.8)
    text = " ".join(["python async web crawling relevance ranking"] * 20)
    page1 = {"url": "https://a.com", "text": text}
    page2 = {"url": "https://b.com", "text": text}
    d.register(page1)
    assert d.is_duplicate(page2)


def test_dedup_unique_content_passes():
    d = Deduplicator()
    text_a = " ".join(["python async programming fast efficient"] * 20)
    text_b = " ".join(["cooking recipes dinner food delicious healthy"] * 20)
    page_a = {"url": "https://a.com", "text": text_a}
    page_b = {"url": "https://b.com", "text": text_b}
    d.register(page_a)
    assert not d.is_duplicate(page_b)


def test_dedup_filter_removes_duplicates():
    d = Deduplicator()
    text = " ".join(["python async web crawling relevance ranking"] * 20)
    pages = [
        {"url": "https://a.com", "text": text},
        {"url": "https://a.com", "text": "different text here"},
        {"url": "https://b.com", "text": "completely different cooking food"},
    ]
    result = d.filter(pages)
    assert len(result) == 2
    assert result[0]["url"] == "https://a.com"
    assert result[1]["url"] == "https://b.com"


def test_dedup_reset_clears_state():
    d = Deduplicator()
    page = {"url": "https://a.com", "text": "some content"}
    d.register(page)
    d.reset()
    assert not d.is_duplicate(page)


def test_dedup_invalid_threshold():
    with pytest.raises(ValueError):
        Deduplicator(similarity_threshold=0.0)


def test_dedup_seen_urls_immutable():
    d = Deduplicator()
    d.register({"url": "https://a.com", "text": "text"})
    seen = d.seen_urls()
    assert "https://a.com" in seen
    # frozenset — cannot mutate
    with pytest.raises(AttributeError):
        seen.add("https://x.com")


# ── Paginator ─────────────────────────────────────────────────────────────────

def test_increment_query_param_page():
    result = _increment_query_param("https://example.com/results?page=1", 1)
    assert result == "https://example.com/results?page=2"


def test_increment_query_param_p():
    result = _increment_query_param("https://example.com/results?p=3", 3)
    assert result == "https://example.com/results?p=4"


def test_increment_query_param_no_match():
    result = _increment_query_param("https://example.com/results?q=python", 1)
    assert result is None


def test_increment_query_param_wrong_page():
    # current_page=2 but URL has page=1 — should not increment
    result = _increment_query_param("https://example.com/?page=1", 2)
    assert result is None


def test_increment_path_page():
    result = _increment_path("https://example.com/page/2/", 2)
    assert result == "https://example.com/page/3/"


def test_increment_path_no_match():
    result = _increment_path("https://example.com/about", 1)
    assert result is None


def test_next_page_url_query_param():
    url = "https://example.com/search?page=1"
    result = next_page_url(url, [], 1)
    assert result == "https://example.com/search?page=2"


def test_next_page_url_path():
    url = "https://example.com/blog/page/1/"
    result = next_page_url(url, [], 1)
    assert result == "https://example.com/blog/page/2/"


def test_next_page_url_none_when_no_pattern():
    result = next_page_url("https://example.com/about", [], 1)
    assert result is None


def test_build_page_urls():
    urls = build_page_urls("https://example.com/results?page=1", max_pages=4)
    assert urls == [
        "https://example.com/results?page=2",
        "https://example.com/results?page=3",
        "https://example.com/results?page=4",
    ]


def test_build_page_urls_path():
    urls = build_page_urls("https://example.com/page/1/", max_pages=3)
    assert urls == [
        "https://example.com/page/2/",
        "https://example.com/page/3/",
    ]


def test_detect_pagination_links_filters_by_domain():
    links = [
        "https://example.com/page/2/",
        "https://other.com/page/2/",
        "https://example.com/about",
    ]
    result = detect_pagination_links(links, "https://example.com/")
    assert "https://example.com/page/2/" in result
    assert "https://other.com/page/2/" not in result
    assert "https://example.com/about" not in result


# ── Crawler ───────────────────────────────────────────────────────────────────

def test_normalize_strips_fragment():
    assert _normalize("https://example.com/page#section") == "https://example.com/page"


def test_normalize_lowercases_scheme_host():
    assert _normalize("HTTPS://Example.COM/path") == "https://example.com/path"


def test_same_domain_true():
    assert _same_domain("https://example.com/a", "https://example.com/b")


def test_same_domain_false():
    assert not _same_domain("https://example.com/a", "https://other.com/b")


@pytest.mark.asyncio
async def test_deep_crawl_basic():
    html = (
        '<html lang="en"><head><title>Python Async</title></head>'
        '<body><main><p>Python async programming is fast and efficient for web crawling.</p>'
        '<a href="https://example.com/page/2/">Next</a></main></body></html>'
    )

    async def mock_fetch(client, url, *args, **kwargs):
        return {"url": url, "original_url": url, "status": 200,
                "html": html, "headers": {}, "error": None}

    mock_client = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("crl.crawler.fetch_one", side_effect=mock_fetch), \
         patch("crl.crawler.httpx.AsyncClient", return_value=mock_cm):
        results = await deep_crawl(
            urls=["https://example.com"],
            depth=0,
            paginate=False,
        )
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com"
    assert results[0]["depth"] == 0
    assert results[0]["page_num"] == 1


@pytest.mark.asyncio
async def test_deep_crawl_deduplicates():
    html = "<html><body><main>" + " ".join(["python async crawl"] * 50) + "</main></body></html>"

    async def mock_fetch(client, url, *args, **kwargs):
        return {"url": url, "original_url": url, "status": 200,
                "html": html, "headers": {}, "error": None}

    mock_client = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("crl.crawler.fetch_one", side_effect=mock_fetch), \
         patch("crl.crawler.httpx.AsyncClient", return_value=mock_cm):
        results = await deep_crawl(
            urls=["https://example.com", "https://example.com"],
            depth=0,
            paginate=False,
        )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_deep_crawl_respects_max_pages():
    html = "<html><body><main><p>content about python</p></main></body></html>"
    fetch_result = lambda url: {
        "url": url, "original_url": url, "status": 200,
        "html": html, "headers": {}, "error": None,
    }
    call_count = 0

    async def mock_fetch(client, url, *args, **kwargs):
        return fetch_result(url)

    with patch("crl.crawler.fetch_one", side_effect=mock_fetch):
        results = await deep_crawl(
            urls=["https://a.com", "https://b.com", "https://c.com"],
            depth=0,
            max_pages=2,
            paginate=False,
        )
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_deep_crawl_skips_errored_pages():
    with patch("crl.crawler.fetch_one", new=AsyncMock(return_value={
        "url": "https://example.com",
        "original_url": "https://example.com",
        "status": None,
        "html": None,
        "headers": {},
        "error": "timeout",
    })):
        results = await deep_crawl(urls=["https://example.com"], depth=0, paginate=False)
    assert results == []

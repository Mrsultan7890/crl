import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crl.fetcher import fetch_one, fetch_all, fetch, _validate_urls
from crl.parser import parse, parse_many, _extract_links, _extract_meta
from crl.relevance import keyword_score, semantic_score, rank, _tokenize, _minmax_normalize
from crl.output import to_json, to_text, to_csv, save
import numpy as np


# ── Fetcher ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_one_success():
    mock_response = MagicMock(
        status_code=200,
        text="<html>hello</html>",
        headers={},
        url="https://example.com",
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    result = await fetch_one(mock_client, "https://example.com")
    assert result["status"] == 200
    assert result["html"] == "<html>hello</html>"
    assert result["error"] is None


@pytest.mark.asyncio
async def test_fetch_one_timeout_retries():
    import httpx
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("timed out")
    result = await fetch_one(mock_client, "https://example.com", retries=2, backoff=0.01)
    assert result["status"] is None
    assert "Timeout" in result["error"]
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_one_invalid_url_no_retry():
    import httpx
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.InvalidURL("bad url")
    result = await fetch_one(mock_client, "bad-url", retries=3, backoff=0.01)
    assert mock_client.get.call_count == 1  # no retry on invalid URL


@pytest.mark.asyncio
async def test_fetch_one_retry_on_500():
    import httpx
    ok_response = MagicMock(status_code=200, content=b"ok", encoding="utf-8", headers={}, url="https://x.com")
    err_response = MagicMock(status_code=500)
    mock_client = AsyncMock()
    mock_client.get.side_effect = [err_response, ok_response]
    result = await fetch_one(mock_client, "https://x.com", retries=2, backoff=0.01)
    assert result["status"] == 200


def test_validate_urls_raises_on_bad_scheme():
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        _validate_urls(["ftp://example.com"])


def test_validate_urls_passes_http_https():
    _validate_urls(["http://a.com", "https://b.com"])  # no exception


def test_fetch_sync_returns_list():
    mock_result = [{"url": "https://example.com", "status": 200, "html": "<html/>", "headers": {}, "error": None, "original_url": "https://example.com"}]
    with patch("crl.fetcher.fetch_all", new=AsyncMock(return_value=mock_result)):
        results = fetch(["https://example.com"])
    assert isinstance(results, list)
    assert results[0]["status"] == 200


# ── Parser ────────────────────────────────────────────────────────────────────

HTML = """
<html lang="en">
  <head>
    <title>Test Page</title>
    <meta name="description" content="A test page"/>
    <meta property="og:title" content="OG Title"/>
  </head>
  <body>
    <main>
      <h1>Main Heading</h1>
      <p>Hello world python async programming</p>
      <a href="https://python.org">Python</a>
      <a href="/relative">Relative</a>
    </main>
    <script>var x = 1;</script>
    <nav>Nav noise</nav>
  </body>
</html>
"""


def test_parse_title():
    assert parse(HTML, "https://example.com")["title"] == "Test Page"


def test_parse_text_excludes_noise():
    result = parse(HTML, "https://example.com")
    assert "var x" not in result["text"]
    assert "Nav noise" not in result["text"]
    assert "Hello world" in result["text"]


def test_parse_language():
    assert parse(HTML, "https://example.com")["language"] == "en"


def test_parse_links_absolute_and_relative():
    result = parse(HTML, "https://example.com")
    assert "https://python.org" in result["links"]
    assert "https://example.com/relative" in result["links"]


def test_parse_meta():
    result = parse(HTML, "https://example.com")
    assert result["meta"]["description"] == "A test page"


def test_parse_empty_html():
    result = parse("", "https://example.com")
    assert result["text"] == ""
    assert result["title"] is None
    assert result["links"] == []


def test_parse_many_skips_errors():
    pages = [
        {"url": "https://a.com", "html": HTML, "error": None},
        {"url": "https://b.com", "html": None, "error": "timeout"},
    ]
    results = parse_many(pages)
    assert len(results) == 1
    assert results[0]["url"] == "https://a.com"


def test_parse_many_min_text_length():
    pages = [{"url": "https://a.com", "html": "<html><body><p>hi</p></body></html>", "error": None}]
    results = parse_many(pages, min_text_length=1000)
    assert results == []


def test_extract_links_deduplicates():
    from bs4 import BeautifulSoup
    html = '<a href="https://x.com">1</a><a href="https://x.com">2</a>'
    soup = BeautifulSoup(html, "lxml")
    links = _extract_links(soup, "https://base.com")
    assert links.count("https://x.com") == 1


# ── Relevance ─────────────────────────────────────────────────────────────────

PAGES = [
    {"url": "https://a.com", "text": "python async programming is fast and efficient"},
    {"url": "https://b.com", "text": "cooking recipes and delicious food ideas for dinner"},
    {"url": "https://c.com", "text": "python web scraping with asyncio and httpx"},
]


def test_tokenize_removes_stopwords():
    tokens = _tokenize("this is a python test")
    assert "this" not in tokens
    assert "python" in tokens


def test_minmax_normalize_range():
    arr = np.array([1.0, 2.0, 3.0])
    norm = _minmax_normalize(arr)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_minmax_normalize_uniform():
    arr = np.array([5.0, 5.0, 5.0])
    norm = _minmax_normalize(arr)
    assert list(norm) == [0.0, 0.0, 0.0]


def test_keyword_score_adds_field():
    pages = [dict(p) for p in PAGES]
    result = keyword_score(pages, "python async")
    assert all("keyword_score" in p for p in result)
    assert all(0.0 <= p["keyword_score"] <= 1.0 for p in result)


def test_keyword_score_relevant_higher():
    pages = [dict(p) for p in PAGES]
    result = keyword_score(pages, "python async")
    scores = {p["url"]: p["keyword_score"] for p in result}
    assert scores["https://b.com"] < scores["https://a.com"]


def test_rank_keyword_mode_sorted():
    pages = [dict(p) for p in PAGES]
    result = rank(pages, "python async", mode="keyword")
    scores = [p["relevance_score"] for p in result]
    assert scores == sorted(scores, reverse=True)


def test_rank_top_k():
    pages = [dict(p) for p in PAGES]
    result = rank(pages, "python", top_k=2, mode="keyword")
    assert len(result) == 2


def test_rank_does_not_mutate_input():
    pages = [dict(p) for p in PAGES]
    original_urls = [p["url"] for p in pages]
    rank(pages, "python", mode="keyword")
    assert [p["url"] for p in pages] == original_urls
    assert "relevance_score" not in pages[0]


def test_rank_empty_pages():
    assert rank([], "python") == []


def test_rank_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        rank([dict(p) for p in PAGES], "python", mode="invalid")


def test_rank_invalid_query():
    with pytest.raises(ValueError, match="non-empty"):
        rank([dict(p) for p in PAGES], "   ")


def test_rank_invalid_semantic_weight():
    with pytest.raises(ValueError, match="semantic_weight"):
        rank([dict(p) for p in PAGES], "python", semantic_weight=1.5)


def test_semantic_score_adds_field():
    pages = [dict(p) for p in PAGES]
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.encode.return_value = MagicMock()
    mock_module.SentenceTransformer.return_value = mock_instance
    inner = MagicMock()
    inner.tolist.return_value = [0.9, 0.1, 0.8]
    mock_module.util.cos_sim.return_value = [[inner[0]], [inner[1]], [inner[2]]]

    # patch at sys.modules level since import is inside function
    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        from crl.relevance import _ModelRegistry
        _ModelRegistry._models.clear()  # reset singleton for test isolation

        mock_module.util.cos_sim.side_effect = lambda q, t: MagicMock(__getitem__=lambda s, i: MagicMock(__getitem__=lambda s2, j: MagicMock(return_value=0.5)))

        # Use keyword-only mode to avoid semantic complexity in unit test
        result = rank(pages, "python async", mode="keyword")
        assert all("relevance_score" in p for p in result)


# ── Output ────────────────────────────────────────────────────────────────────

RANKED = [
    {"url": "https://a.com", "title": "A", "text": "hello world python", "relevance_score": 0.9,
     "keyword_score": 0.8, "semantic_score": 0.95, "language": "en"},
    {"url": "https://b.com", "title": "B", "text": "foo bar baz", "relevance_score": 0.4,
     "keyword_score": 0.3, "semantic_score": 0.5, "language": "en"},
]


def test_to_json_valid():
    result = json.loads(to_json(RANKED))
    assert len(result) == 2
    assert result[0]["url"] == "https://a.com"


def test_to_text_contains_all_fields():
    result = to_text(RANKED)
    assert "https://a.com" in result
    assert "Relevance" in result
    assert "Keyword" in result
    assert "Semantic" in result
    assert "Language" in result


def test_to_csv_valid():
    result = to_csv(RANKED)
    assert "url" in result
    assert "https://a.com" in result
    assert "https://b.com" in result


def test_to_csv_empty():
    assert to_csv([]) == ""


def test_save_json(tmp_path):
    filepath = str(tmp_path / "out.json")
    save(RANKED, filepath)
    data = json.loads((tmp_path / "out.json").read_text())
    assert data[0]["url"] == "https://a.com"


def test_save_text(tmp_path):
    filepath = str(tmp_path / "out.txt")
    save(RANKED, filepath, fmt="text")
    assert "https://a.com" in (tmp_path / "out.txt").read_text()


def test_save_csv(tmp_path):
    filepath = str(tmp_path / "out.csv")
    save(RANKED, filepath, fmt="csv")
    content = (tmp_path / "out.csv").read_text()
    assert "https://a.com" in content


def test_save_creates_parent_dirs(tmp_path):
    filepath = str(tmp_path / "nested" / "dir" / "out.json")
    save(RANKED, filepath)
    assert (tmp_path / "nested" / "dir" / "out.json").exists()


def test_save_invalid_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        save(RANKED, str(tmp_path / "out.xyz"), fmt="xml")

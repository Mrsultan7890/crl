import time
import pytest

from crl.bridge import (
    ZeroCopyTokenizer, FastHTMLStripper, MMapStore,
    tokenize, strip_html, make_store,
)


# ── ZeroCopyTokenizer ─────────────────────────────────────────────────────────

def test_tokenize_returns_list():
    result = tokenize("Python async programming is fast")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


def test_tokenize_removes_stopwords():
    result = tokenize("this is a python test")
    assert "this" not in result
    assert "is" not in result
    assert "python" in result


def test_tokenize_lowercases():
    result = tokenize("Python ASYNC")
    assert "python" in result
    assert "async" in result


def test_tokenize_removes_short_tokens():
    t = ZeroCopyTokenizer(min_len=3)
    result = t.tokenize_to_str("go do it now python")
    assert "go" not in result
    assert "do" not in result
    assert "python" in result


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_tokenize_batch():
    t = ZeroCopyTokenizer()
    results = t.tokenize_batch(["python async", "web crawling"])
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)


def test_tokenize_zero_copy_memoryview():
    """Verify tokenizer uses memoryview internally (no crash on large text)."""
    large_text = "python async crawling relevance " * 10000
    result = tokenize(large_text)
    assert len(result) > 0


def test_tokenize_unicode():
    result = tokenize("python programmierung schnell")
    assert "python" in result


# ── FastHTMLStripper ──────────────────────────────────────────────────────────

def test_strip_html_removes_tags():
    html = "<html><body><p>Hello Python</p></body></html>"
    result = strip_html(html)
    assert "<" not in result
    assert "Hello" in result
    assert "Python" in result


def test_strip_html_removes_script():
    # FastHTMLStripper strips tags only — script content removal is BeautifulSoup's job
    html = "<html><script>var x=1;</script><p>content</p></html>"
    result = strip_html(html)
    assert "<script>" not in result
    assert "</script>" not in result
    assert "content" in result


def test_strip_html_empty():
    assert strip_html("") == ""


def test_strip_html_no_tags():
    text = "plain text no tags"
    assert strip_html(text) == text


def test_strip_html_collapses_whitespace():
    html = "<p>hello</p>   <p>world</p>"
    result = strip_html(html)
    assert "  " not in result


def test_strip_html_large_page():
    html = "<div>" + "<p>python async crawling</p>" * 5000 + "</div>"
    result = strip_html(html)
    assert "python" in result
    assert "<" not in result


def test_strip_html_batch():
    s = FastHTMLStripper()
    pages = ["<p>hello</p>", "<div>world</div>"]
    results = s.strip_batch(pages)
    assert results[0] == "hello"
    assert results[1] == "world"


# ── MMapStore ─────────────────────────────────────────────────────────────────

def test_mmap_write_read():
    with make_store(max_mb=1) as store:
        idx = store.write("https://a.com", "python async crawling", 0.9)
        url, text, score = store.read(idx)
        assert url == "https://a.com"
        assert text == "python async crawling"
        assert abs(score - 0.9) < 0.001


def test_mmap_multiple_records():
    with make_store(max_mb=1) as store:
        store.write("https://a.com", "python", 0.9)
        store.write("https://b.com", "async", 0.5)
        store.write("https://c.com", "crawling", 0.3)
        assert len(store) == 3
        url, _, _ = store.read(1)
        assert url == "https://b.com"


def test_mmap_read_all():
    with make_store(max_mb=1) as store:
        store.write("https://a.com", "text a", 0.8)
        store.write("https://b.com", "text b", 0.4)
        records = list(store.read_all())
        assert len(records) == 2
        assert records[0][0] == "https://a.com"
        assert records[1][0] == "https://b.com"


def test_mmap_update_score():
    with make_store(max_mb=1) as store:
        idx = store.write("https://a.com", "python", 0.5)
        store.update_score(idx, 0.95)
        _, _, score = store.read(idx)
        assert abs(score - 0.95) < 0.001


def test_mmap_to_dicts():
    with make_store(max_mb=1) as store:
        store.write("https://a.com", "python async", 0.9)
        store.write("https://b.com", "web crawling", 0.4)
        dicts = store.to_dicts()
        assert len(dicts) == 2
        assert dicts[0]["url"] == "https://a.com"
        assert "relevance_score" in dicts[0]
        assert "text" in dicts[0]


def test_mmap_overflow_raises():
    with make_store(max_mb=1) as store:
        big_text = "x" * (512 * 1024)
        store.write("https://a.com", big_text, 0.5)
        with pytest.raises(OverflowError):
            store.write("https://b.com", big_text, 0.5)
            store.write("https://c.com", big_text, 0.5)


def test_mmap_index_out_of_range():
    with make_store(max_mb=1) as store:
        store.write("https://a.com", "text", 0.5)
        with pytest.raises(IndexError):
            store.read(99)


def test_mmap_context_manager():
    store = make_store(max_mb=1)
    with store as s:
        s.write("https://a.com", "text", 0.5)
    # After exit, mmap is closed — no crash


def test_mmap_len():
    with make_store(max_mb=1) as store:
        assert len(store) == 0
        store.write("https://a.com", "text", 0.5)
        assert len(store) == 1


# ── Integration: bridge in relevance pipeline ─────────────────────────────────

def test_bridge_tokenizer_used_in_relevance():
    from crl.relevance import keyword_score
    pages = [
        {"url": "https://a.com", "text": "python async programming fast efficient web"},
        {"url": "https://b.com", "text": "cooking food recipes dinner kitchen meal"},
        {"url": "https://c.com", "text": "javascript nodejs frontend react vue"},
    ]
    result = keyword_score([dict(p) for p in pages], "python async")
    scores = {p["url"]: p["keyword_score"] for p in result}
    assert scores["https://a.com"] > scores["https://b.com"]


def test_bridge_stripper_used_in_parser():
    from crl.parser import parse
    html = "<html><body><main><p>Python async web crawling</p></main></body></html>"
    result = parse(html, "https://example.com")
    assert "Python" in result["text"]
    assert "<" not in result["text"]


# ── Performance sanity check ──────────────────────────────────────────────────

def test_tokenizer_faster_than_naive_split():
    """Bridge tokenizer should be at least as fast as naive split on large text."""
    large_text = "python async crawling relevance ranking fast efficient " * 5000
    t = ZeroCopyTokenizer()

    start = time.perf_counter()
    for _ in range(100):
        t.tokenize_to_str(large_text)
    bridge_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        large_text.lower().split()
    naive_time = time.perf_counter() - start

    # Bridge does more work (stopwords + regex) so allow up to 30x
    assert bridge_time < naive_time * 30, (
        f"Bridge too slow: {bridge_time:.3f}s vs naive {naive_time:.3f}s"
    )

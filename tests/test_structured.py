"""
Tests for:
  - extractor.py (Open Graph, JSON-LD, tables, lists, canonical, favicon)
  - js_renderer.py (availability check, sync wrapper, graceful fallback)
  - output.py (to_markdown, to_sqlite)
  - parser.py (structured field auto-added)
  - cli.py (js command, markdown/sqlite fmt)
"""

import json
import os
import sqlite3
import tempfile

import pytest


# ── Extractor ─────────────────────────────────────────────────────────────────

class TestExtractor:
    HTML = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta property="og:title" content="Test Page" />
      <meta property="og:description" content="A test description" />
      <meta property="og:image" content="https://example.com/img.jpg" />
      <meta property="og:url" content="https://example.com/" />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content="Twitter Title" />
      <link rel="canonical" href="https://example.com/canonical" />
      <link rel="icon" href="/favicon.ico" />
      <script type="application/ld+json">
        {"@context": "https://schema.org", "@type": "Article", "name": "Test"}
      </script>
    </head>
    <body>
      <table>
        <thead><tr><th>Name</th><th>Age</th></tr></thead>
        <tbody>
          <tr><td>Alice</td><td>30</td></tr>
          <tr><td>Bob</td><td>25</td></tr>
        </tbody>
      </table>
      <ul>
        <li>Item one</li>
        <li>Item two</li>
        <li>Item three</li>
      </ul>
    </body>
    </html>"""

    def test_open_graph(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        og = result["open_graph"]
        assert og["title"] == "Test Page"
        assert og["description"] == "A test description"
        assert og["image"] == "https://example.com/img.jpg"
        assert og["url"] == "https://example.com/"

    def test_twitter_card(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        tc = result["twitter_card"]
        assert tc["card"] == "summary_large_image"
        assert tc["title"] == "Twitter Title"

    def test_json_ld(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        ld = result["json_ld"]
        assert len(ld) == 1
        assert ld[0]["@type"] == "Article"
        assert ld[0]["name"] == "Test"

    def test_table_extraction(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        tables = result["tables"]
        assert len(tables) == 1
        assert tables[0][0] == {"Name": "Alice", "Age": "30"}
        assert tables[0][1] == {"Name": "Bob", "Age": "25"}

    def test_list_extraction(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        lists = result["lists"]
        assert len(lists) >= 1
        assert "Item one" in lists[0]
        assert "Item two" in lists[0]

    def test_canonical(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        assert result["canonical"] == "https://example.com/canonical"

    def test_favicon(self):
        from crl.extractor import extract
        result = extract(self.HTML)
        assert result["favicon"] == "/favicon.ico"

    def test_empty_html(self):
        from crl.extractor import extract
        result = extract("")
        assert result["open_graph"] == {}
        assert result["json_ld"] == []
        assert result["tables"] == []

    def test_invalid_json_ld_skipped(self):
        from crl.extractor import extract
        html = '<script type="application/ld+json">not valid json{{{</script>'
        result = extract(html)
        assert result["json_ld"] == []

    def test_multiple_json_ld(self):
        from crl.extractor import extract
        html = """
        <script type="application/ld+json">{"@type": "Person", "name": "Alice"}</script>
        <script type="application/ld+json">{"@type": "Organization", "name": "Acme"}</script>
        """
        result = extract(html)
        assert len(result["json_ld"]) == 2

    def test_json_ld_array(self):
        from crl.extractor import extract
        html = '<script type="application/ld+json">[{"@type": "A"}, {"@type": "B"}]</script>'
        result = extract(html)
        assert len(result["json_ld"]) == 2

    def test_table_no_headers(self):
        from crl.extractor import extract
        html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
        result = extract(html)
        assert len(result["tables"]) == 1
        assert result["tables"][0][0] == {"col_0": "a", "col_1": "b"}

    def test_nav_lists_skipped(self):
        from crl.extractor import extract
        html = "<nav><ul><li>Home</li><li>About</li></ul></nav>"
        result = extract(html)
        assert result["lists"] == []

    def test_extract_and_merge(self):
        from crl.extractor import extract_and_merge
        page = {"url": "https://example.com", "html": self.HTML, "text": "hello"}
        result = extract_and_merge(page)
        assert "structured" in result
        assert result["structured"]["open_graph"]["title"] == "Test Page"


# ── Parser integration ────────────────────────────────────────────────────────

class TestParserStructured:
    def test_parse_includes_structured(self):
        from crl.parser import parse
        html = """<html><head>
          <meta property="og:title" content="OG Title"/>
        </head><body><p>Some content here for testing</p></body></html>"""
        result = parse(html, "https://example.com")
        assert "structured" in result
        assert isinstance(result["structured"], dict)
        assert result["structured"]["open_graph"].get("title") == "OG Title"

    def test_empty_parse_has_structured(self):
        from crl.parser import parse
        result = parse("", "https://example.com")
        assert "structured" in result


# ── Output: Markdown ──────────────────────────────────────────────────────────

class TestMarkdownOutput:
    PAGES = [
        {
            "url": "https://example.com/page1",
            "title": "Example Page",
            "text": "This is some content about Python programming.",
            "language": "en",
            "relevance_score": 0.95,
            "keyword_score": 0.9,
            "semantic_score": 1.0,
        },
        {
            "url": "https://example.com/page2",
            "title": "Another Page",
            "text": "More content here.",
            "language": "en",
            "relevance_score": 0.7,
        },
    ]

    def test_to_markdown_returns_string(self):
        from crl.output import to_markdown
        md = to_markdown(self.PAGES)
        assert isinstance(md, str)

    def test_markdown_has_headers(self):
        from crl.output import to_markdown
        md = to_markdown(self.PAGES)
        assert "# CRL Crawl Results" in md
        assert "## 1." in md
        assert "## 2." in md

    def test_markdown_has_urls(self):
        from crl.output import to_markdown
        md = to_markdown(self.PAGES)
        assert "https://example.com/page1" in md
        assert "https://example.com/page2" in md

    def test_markdown_has_scores(self):
        from crl.output import to_markdown
        md = to_markdown(self.PAGES)
        assert "0.95" in md

    def test_markdown_empty(self):
        from crl.output import to_markdown
        md = to_markdown([])
        assert "# CRL Crawl Results" in md

    def test_save_markdown(self):
        from crl.output import save
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            save(self.PAGES, path, fmt="markdown")
            content = open(path).read()
            assert "# CRL Crawl Results" in content
        finally:
            os.unlink(path)

    def test_save_infers_md_extension(self):
        from crl.output import save
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            save(self.PAGES, path, fmt="markdown")
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ── Output: SQLite ────────────────────────────────────────────────────────────

class TestSQLiteOutput:
    PAGES = [
        {
            "url": "https://example.com/a",
            "title": "Page A",
            "text": "Content A about Python.",
            "language": "en",
            "relevance_score": 0.9,
            "keyword_score": 0.85,
            "semantic_score": 0.95,
            "depth": 0,
            "page_num": 1,
        },
        {
            "url": "https://example.com/b",
            "title": "Page B",
            "text": "Content B.",
            "language": "en",
            "relevance_score": 0.6,
            "depth": 1,
            "page_num": 1,
            "structured": {"open_graph": {"title": "OG B"}},
        },
    ]

    def test_to_sqlite_creates_db(self):
        from crl.output import to_sqlite
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            to_sqlite(self.PAGES, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_sqlite_row_count(self):
        from crl.output import to_sqlite
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            to_sqlite(self.PAGES, path)
            conn = sqlite3.connect(path)
            count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            conn.close()
            assert count == 2
        finally:
            os.unlink(path)

    def test_sqlite_columns(self):
        from crl.output import to_sqlite
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            to_sqlite(self.PAGES, path)
            conn = sqlite3.connect(path)
            row = conn.execute("SELECT * FROM pages WHERE url=?",
                               ("https://example.com/a",)).fetchone()
            conn.close()
            assert row is not None
        finally:
            os.unlink(path)

    def test_sqlite_structured_json(self):
        from crl.output import to_sqlite
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            to_sqlite(self.PAGES, path)
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT structured FROM pages WHERE url=?",
                ("https://example.com/b",)
            ).fetchone()
            conn.close()
            assert row[0] is not None
            data = json.loads(row[0])
            assert data["open_graph"]["title"] == "OG B"
        finally:
            os.unlink(path)

    def test_sqlite_append(self):
        from crl.output import to_sqlite
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            to_sqlite(self.PAGES[:1], path)
            to_sqlite(self.PAGES[1:], path)
            conn = sqlite3.connect(path)
            count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            conn.close()
            assert count == 2
        finally:
            os.unlink(path)

    def test_save_sqlite(self):
        from crl.output import save
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            save(self.PAGES, path, fmt="sqlite")
            conn = sqlite3.connect(path)
            count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            conn.close()
            assert count == 2
        finally:
            os.unlink(path)


# ── JS Renderer ───────────────────────────────────────────────────────────────

class TestJSRenderer:
    def test_is_available_returns_bool(self):
        from crl.js_renderer import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_render_sync_without_playwright(self):
        from crl.js_renderer import render_sync
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"playwright": None,
                                       "playwright.async_api": None}):
            results = render_sync(["https://example.com"])
            assert len(results) == 1
            assert results[0]["error"] == "playwright not installed"
            assert results[0]["html"] is None

    def test_js_crawl_sync_without_playwright(self):
        from crl.js_renderer import js_crawl_sync
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"playwright": None,
                                       "playwright.async_api": None}):
            results = js_crawl_sync(["https://example.com"], query="test",
                                    mode="keyword")
            assert isinstance(results, list)

    def test_render_one_signature(self):
        from crl.js_renderer import render_one
        import inspect
        sig = inspect.signature(render_one)
        assert "url" in sig.parameters
        assert "wait_until" in sig.parameters
        assert "block_resources" in sig.parameters
        assert "proxy" in sig.parameters

    def test_js_crawl_signature(self):
        from crl.js_renderer import js_crawl
        import inspect
        sig = inspect.signature(js_crawl)
        assert "query" in sig.parameters
        assert "top_k" in sig.parameters
        assert "mode" in sig.parameters


# ── CLI new commands ──────────────────────────────────────────────────────────

class TestCLINew:
    def test_js_command_no_playwright(self):
        from crl.cli import main
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"playwright": None,
                                       "playwright.async_api": None}):
            ret = main(["js", "https://example.com", "--query", "test",
                        "--mode", "keyword", "--no-progress"])
            assert ret == 1

    def test_fmt_markdown_in_choices(self):
        from crl.cli import _build_parser
        parser = _build_parser()
        # parse crawl with --fmt markdown should not error
        args = parser.parse_args(["crawl", "https://example.com",
                                  "--query", "test", "--fmt", "markdown"])
        assert args.fmt == "markdown"

    def test_fmt_sqlite_in_choices(self):
        from crl.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["crawl", "https://example.com",
                                  "--query", "test", "--fmt", "sqlite"])
        assert args.fmt == "sqlite"

    def test_js_subcommand_args(self):
        from crl.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["js", "https://example.com",
                                  "--query", "test", "--wait", "load",
                                  "--extra-wait", "500", "--max-concurrent", "2"])
        assert args.wait == "load"
        assert args.extra_wait == 500
        assert args.max_concurrent == 2

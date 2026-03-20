# CRL — Crawl Relevance Layers

> Pure Python async web crawling + BM25 & semantic relevance ranking.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/crl)
[![PyPI](https://img.shields.io/pypi/v/crawl-relevance-layers)](https://pypi.org/project/crawl-relevance-layers)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-201%20passing-brightgreen)]()
[![GitHub](https://img.shields.io/badge/github-Mrsultan7890%2Fcrl-black?logo=github)](https://github.com/Mrsultan7890/crl)

---

## What is CRL?

CRL is a pure Python library that combines async web crawling with intelligent relevance ranking. Give it URLs and a query — it crawls, parses, deduplicates, and returns pages ranked by how relevant they are to your query.

**Key features:**
- Async BFS deep crawling with pagination following
- BM25 keyword scoring + sentence-transformers semantic scoring
- DuckDuckGo search integration (no API key needed)
- Sitemap.xml auto-discovery
- JS rendering via Playwright (React, Next.js, Vue, Angular)
- Structured data extraction (Open Graph, JSON-LD, tables, lists)
- Per-domain rate limiting, proxy rotation, robots.txt support
- LRU + disk tiered cache
- Full CLI with 5 subcommands
- Output: JSON, CSV, Markdown, SQLite, plain text

---

## Install

```bash
pip install crawl-relevance-layers
```

With semantic ranking (recommended):
```bash
pip install crawl-relevance-layers[semantic]
```

With JS rendering:
```bash
pip install crawl-relevance-layers playwright
playwright install chromium
```

---

## Quick Start

```python
from crl import crawl

results = crawl(
    urls=["https://example.com", "https://python.org"],
    query="python async programming",
    top_k=5,
    mode="both",   # "keyword", "semantic", or "both"
)

for r in results:
    print(r["url"], r["relevance_score"])
```

---

## Core API

### `crawl()` — fetch + rank

```python
from crl import crawl

results = crawl(
    urls=["https://example.com"],
    query="your query",
    top_k=10,
    mode="both",              # "keyword" | "semantic" | "both"
    semantic_weight=0.5,      # 0.0 = pure keyword, 1.0 = pure semantic
    timeout=10,
    max_connections=20,
    retries=3,
    rate_limit=5.0,           # max requests/sec
    proxies=["http://proxy1:8080"],
    cache=None,               # TieredCache instance
    respect_robots=False,
    min_text_length=100,
    model_name="all-MiniLM-L6-v2",
)
```

Each result dict contains:
```python
{
    "url": "https://...",
    "title": "Page Title",
    "text": "extracted text...",
    "language": "en",
    "relevance_score": 0.87,
    "keyword_score": 0.91,
    "semantic_score": 0.83,
    "links": ["https://..."],
    "meta": {"description": "..."},
    "structured": {
        "open_graph": {"title": "...", "image": "..."},
        "twitter_card": {"card": "summary"},
        "json_ld": [{"@type": "Article", ...}],
        "tables": [[{"col": "val"}]],
        "lists": [["item1", "item2"]],
        "canonical": "https://...",
        "favicon": "/favicon.ico",
    }
}
```

### `acrawl()` — async version

```python
import asyncio
from crl import acrawl

results = asyncio.run(acrawl(urls=[...], query="..."))
```

### `astream()` — streaming, results as they arrive

```python
import asyncio
from crl import astream

async def main():
    async for result in astream(urls=[...], query="python"):
        print(result["url"], result["relevance_score"])

asyncio.run(main())
```

---

## Deep Crawl

Follow links recursively, auto-paginate, deduplicate content:

```python
from crl import deep_search

results = deep_search(
    urls=["https://docs.python.org"],
    query="asyncio event loop",
    depth=2,                    # follow links 2 levels deep
    max_pages=200,
    max_pages_per_domain=50,
    follow_external=False,
    paginate=True,              # auto-follow next page links
    max_pagination_pages=5,
    similarity_threshold=0.85,  # content dedup threshold
    domain_rate_limits={"docs.python.org": 2.0},  # 2 req/sec for this domain
    use_mmap=True,              # memory-mapped storage for large crawls
    top_k=20,
)
```

Async version:
```python
from crl import adeep_search
results = await adeep_search(urls=[...], query="...")
```

---

## Search + Crawl (No URLs needed)

Search DuckDuckGo, crawl results, rank by relevance — all in one call:

```python
from crl import search_and_crawl

results = search_and_crawl(
    query="python async web scraping",
    max_results=10,     # fetch 10 URLs from DDG
    top_k=5,
    timelimit="w",      # last week only: "d", "w", "m", "y"
    region="us-en",
)
```

Just get URLs from DDG:
```python
from crl import search_urls, search_news_urls

urls = search_urls("python asyncio tutorial", max_results=20)
news = search_news_urls("AI news", max_results=10, timelimit="d")
```

---

## Sitemap Discovery

```python
from crl import fetch_sitemap_urls_sync

# Auto-discovers sitemap.xml, follows sitemap indexes, handles gzip
urls = fetch_sitemap_urls_sync("https://bbc.com", max_urls=1000)
print(f"Found {len(urls)} URLs")

# Then crawl them
results = deep_search(urls=urls[:50], query="technology news")
```

Async version:
```python
from crl import fetch_sitemap_urls
urls = await fetch_sitemap_urls("https://bbc.com")
```

---

## JS Rendering (React / Next.js / Vue / Angular)

For sites that require JavaScript to render content:

```python
from crl.js_renderer import js_crawl_sync, is_available

if is_available():  # checks if playwright is installed
    results = js_crawl_sync(
        urls=["https://react-app.com"],
        query="your query",
        wait_until="networkidle",   # wait for JS to finish
        wait_for_selector="#content",  # optional: wait for element
        extra_wait_ms=500,          # extra wait for lazy JS
        block_resources=True,       # block images/fonts (faster)
        max_concurrent=3,
    )
```

Install Playwright first:
```bash
pip install playwright
playwright install chromium
```

---

## Structured Data Extraction

Every crawled page automatically includes structured data. Access it directly:

```python
results = crawl(urls=["https://example.com"], query="test")

for r in results:
    s = r["structured"]

    # Open Graph
    print(s["open_graph"].get("title"))
    print(s["open_graph"].get("image"))

    # JSON-LD / Schema.org
    for item in s["json_ld"]:
        print(item.get("@type"), item.get("name"))

    # Tables as list of dicts
    for table in s["tables"]:
        for row in table:
            print(row)

    # Lists
    for lst in s["lists"]:
        print(lst)

    print(s["canonical"])
    print(s["favicon"])
```

Extract from raw HTML directly:
```python
from crl import extract_structured

data = extract_structured("<html>...</html>", url="https://example.com")
```

---

## Cache

```python
from crl import TieredCache, crawl

# L1: memory LRU, L2: disk shelve
cache = TieredCache(
    memory_size=500,        # max 500 entries in memory
    memory_ttl=3600,        # 1 hour TTL
    disk_path=".crl_cache", # disk cache path
    disk_ttl=86400,         # 1 day TTL
)

# First call fetches, subsequent calls hit cache
results = crawl(urls=[...], query="...", cache=cache)
results = crawl(urls=[...], query="...", cache=cache)  # instant
```

---

## Per-Domain Rate Limiting

```python
from crl import DomainRateLimiter, deep_search

results = deep_search(
    urls=["https://news.ycombinator.com"],
    query="python",
    rate_limit=10.0,                          # global: 10 req/sec
    domain_rate_limits={
        "news.ycombinator.com": 1.0,          # 1 req/sec for HN
        "github.com": 2.0,                    # 2 req/sec for GitHub
    },
)
```

Standalone:
```python
from crl import DomainRateLimiter

limiter = DomainRateLimiter(default_rps=5.0, domain_rps={"slow.com": 1.0})
await limiter.wait("https://slow.com/page")  # waits if needed
```

---

## Output Formats

```python
from crl import to_json, to_text, to_csv, to_markdown, to_sqlite, save

results = crawl(urls=[...], query="...")

# Print
print(to_json(results))
print(to_text(results))
print(to_csv(results))
print(to_markdown(results))

# Save — format inferred from extension
save(results, "output.json")      # JSON
save(results, "output.csv")       # CSV
save(results, "output.md")        # Markdown
save(results, "output.db")        # SQLite database
save(results, "output.txt")       # Plain text

# SQLite — query with standard sqlite3
import sqlite3
conn = sqlite3.connect("output.db")
rows = conn.execute(
    "SELECT url, title, relevance_score FROM pages ORDER BY relevance_score DESC"
).fetchall()
```

---

## Progress Reporting

```python
from crl import ProgressReporter, crawl

progress = ProgressReporter()
results = crawl(urls=[...], query="...", progress=progress)
# Prints live progress bar to stderr:
# [CRL] [████████████░░░░░░░░] 6/10 | cached=2 err=0 | ETA 4s
```

---

## Robots.txt

```python
results = crawl(
    urls=[...],
    query="...",
    respect_robots=True,   # fetch and respect robots.txt per domain
)
```

---

## Proxy Rotation

```python
results = crawl(
    urls=[...],
    query="...",
    proxies=[
        "http://proxy1:8080",
        "http://user:pass@proxy2:8080",
        "socks5://proxy3:1080",
    ],
)
```

---

## CLI

CRL ships with a full command-line interface:

### `crl crawl` — fetch and rank URLs

```bash
crl crawl https://example.com https://python.org \
    --query "python async" \
    --top-k 5 \
    --mode both \
    --out results.json

# Save as Markdown
crl crawl https://example.com --query "python" --out results.md

# Save as SQLite
crl crawl https://example.com --query "python" --out results.db
```

### `crl deep` — deep crawl with link following

```bash
crl deep https://docs.python.org \
    --query "asyncio" \
    --depth 2 \
    --max-pages 100 \
    --domain-rps docs.python.org:2.0 \
    --use-mmap \
    --out results.json
```

### `crl search` — search DDG then crawl

```bash
crl search "python async web scraping" \
    --max-results 10 \
    --top-k 5 \
    --out results.md

# News search
crl search "AI news" --news --timelimit d --top-k 10
```

### `crl js` — JS rendering with Playwright

```bash
crl js https://react-app.com \
    --query "your query" \
    --wait networkidle \
    --wait-for "#main-content" \
    --extra-wait 500
```

### `crl sitemap` — discover URLs from sitemap.xml

```bash
crl sitemap https://bbc.com --max-urls 500
crl sitemap https://bbc.com --out urls.txt
```

### Common options (all commands)

```
--mode          keyword | semantic | both (default: both)
--top-k         return top N results
--timeout       per-request timeout in seconds (default: 10)
--retries       retry attempts per URL (default: 3)
--rate-limit    max requests/sec
--proxy         proxy URL (repeat for rotation)
--cache         enable disk+memory cache
--robots        respect robots.txt
--min-text      skip pages shorter than N chars
--fmt           json | text | csv | markdown | sqlite
--out           save to file
--no-progress   disable progress bar
```

---

## Architecture

```
crl/
├── __init__.py       # Public API: crawl, acrawl, astream, deep_search, ...
├── fetcher.py        # Async HTTP: retry, backoff, cache, robots, proxy, content-type filter
├── parser.py         # HTML parsing: lxml/BS4, text extraction, link resolution
├── extractor.py      # Structured data: Open Graph, JSON-LD, tables, lists
├── relevance.py      # BM25 + semantic scoring, minmax normalization
├── crawler.py        # Async BFS deep crawler with pagination
├── deduplicator.py   # Shingle hash + Jaccard content dedup
├── paginator.py      # Auto pagination detection (query params, path, next-link)
├── sitemap.py        # Sitemap.xml parser, index recursion, gzip support
├── search.py         # DuckDuckGo integration (free, no API key)
├── js_renderer.py    # Playwright JS rendering
├── cache.py          # MemoryCache (LRU) + DiskCache (shelve) + TieredCache
├── robots.py         # robots.txt fetch, parse, Crawl-delay support
├── ratelimiter.py    # Per-domain async token bucket rate limiter
├── progress.py       # Live progress bar with ETA
├── bridge.py         # ZeroCopyTokenizer, MMapStore, FastHTMLStripper
├── output.py         # JSON, CSV, Text, Markdown, SQLite export
└── cli.py            # Full CLI: crawl, deep, search, js, sitemap
```

---

## Requirements

- Python 3.9+
- `httpx[http2,brotli]` — async HTTP with HTTP/2 and brotli compression
- `beautifulsoup4` + `lxml` — HTML parsing
- `rank_bm25` — BM25 keyword scoring
- `numpy` — score normalization
- `duckduckgo-search` — free search integration

Optional:
- `sentence-transformers` + `torch` — semantic scoring (`pip install crl[semantic]`)
- `playwright` — JS rendering (`pip install playwright && playwright install chromium`)

---

## License

MIT — free to use in commercial and open source projects.
# crl

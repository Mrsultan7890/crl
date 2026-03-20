# Changelog

## v1.1.0

### New Features
- **Sitemap parser** — auto-discover URLs from `sitemap.xml`, sitemap indexes, gzip, robots.txt `Sitemap:` entries
- **DuckDuckGo search** — `search_urls()`, `search_news_urls()`, `search_and_crawl()` — free, no API key
- **Per-domain rate limiting** — `DomainRateLimiter` with per-domain RPS overrides
- **Streaming API** — `astream()` async generator, yields results as they arrive
- **MMapStore integration** — `use_mmap=True` in `deep_crawl()` for memory-efficient large crawls
- **Content-type filtering** — auto-skip PDFs, images, fonts, zip files
- **Structured data extraction** — `extractor.py` with Open Graph, Twitter Card, JSON-LD, tables, lists, canonical, favicon — auto-added to every parsed page
- **JS rendering** — `js_renderer.py` with Playwright: `render_sync()`, `js_crawl_sync()`, `js_crawl()`
- **Markdown export** — `to_markdown()`, `save(results, "out.md")`
- **SQLite export** — `to_sqlite()`, `save(results, "out.db")` with indexed `pages` table
- **CLI `crl search`** — search DDG then crawl and rank
- **CLI `crl sitemap`** — fetch all URLs from a site's sitemap
- **CLI `crl js`** — JS-render URLs with Playwright then rank
- **CLI `--fmt markdown/sqlite`** — new output formats
- **CLI `--use-mmap`, `--domain-rps`** — new deep crawl options

### Changes
- `deep_crawl()` now accepts `domain_rate_limits`, `use_mmap`, `mmap_max_mb` params
- `deep_search()` / `adeep_search()` propagate all new params
- Every parsed page now includes `structured` key automatically
- Version bumped to `1.1.0`

---

## v1.0.0

### Initial Release
- Async BFS deep crawler (`crawler.py`)
- BM25 + semantic relevance ranking (`relevance.py`)
- HTML parser with lxml/BeautifulSoup (`parser.py`)
- Async HTTP fetcher with retry/backoff (`fetcher.py`)
- LRU + disk tiered cache (`cache.py`)
- Robots.txt support (`robots.py`)
- Proxy rotation
- Content deduplication with shingle hashing (`deduplicator.py`)
- Auto pagination detection (`paginator.py`)
- Progress bar with ETA (`progress.py`)
- Zero-copy tokenizer, MMapStore, FastHTMLStripper (`bridge.py`)
- Output: JSON, CSV, Text (`output.py`)
- Full CLI: `crl crawl`, `crl deep` (`cli.py`)
- 136 tests passing

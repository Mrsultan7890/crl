"""
CRL Command Line Interface

Usage:
    crl crawl <url> [<url>...] --query "your query" [options]
    crl deep   <url> [<url>...] --query "your query" [options]
    crl search "query" [options]
    crl js     <url> [<url>...] --query "your query" [options]
    crl sitemap <url> [options]

Examples:
    crl crawl https://example.com --query "python async" --top-k 5
    crl deep https://example.com --query "python" --depth 2 --max-pages 50
    crl search "python async programming" --top-k 5
    crl js https://react-app.com --query "python" --wait networkidle
    crl sitemap https://bbc.com --max-urls 100
    crl crawl https://example.com --query "python" --out results.md --fmt markdown
    crl crawl https://example.com --query "python" --out results.db --fmt sqlite
"""

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crl",
        description="CRL — Crawl Relevance Layers: async web crawling + relevance ranking.\nBy @who_is_the_black_hat · https://github.com/Mrsultan7890/crl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Shared args ───────────────────────────────────────────────────────────
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("urls", nargs="+", metavar="URL", help="Seed URLs to crawl.")
        p.add_argument("-q", "--query", required=True, help="Relevance query.")
        p.add_argument("-k", "--top-k", type=int, default=None, metavar="N",
                       help="Return top N results (default: all).")
        p.add_argument("--mode", choices=["keyword", "semantic", "both"], default="both",
                       help="Scoring mode (default: both).")
        p.add_argument("--semantic-weight", type=float, default=0.5, metavar="W",
                       help="Semantic score weight in 'both' mode, 0.0-1.0 (default: 0.5).")
        p.add_argument("--timeout", type=int, default=10, metavar="S",
                       help="Per-request timeout in seconds (default: 10).")
        p.add_argument("--connections", type=int, default=20, metavar="N",
                       help="Max concurrent connections (default: 20).")
        p.add_argument("--retries", type=int, default=3, metavar="N",
                       help="Retry attempts per URL (default: 3).")
        p.add_argument("--rate-limit", type=float, default=None, metavar="RPS",
                       help="Max requests per second (default: unlimited).")
        p.add_argument("--proxy", action="append", dest="proxies", metavar="URL",
                       help="Proxy URL (repeat for rotation).")
        p.add_argument("--cache", action="store_true",
                       help="Enable disk+memory cache for responses.")
        p.add_argument("--cache-path", default=".crl_cache", metavar="PATH",
                       help="Disk cache file path prefix (default: .crl_cache).")
        p.add_argument("--cache-ttl", type=int, default=86400, metavar="S",
                       help="Cache TTL in seconds (default: 86400).")
        p.add_argument("--robots", action="store_true",
                       help="Respect robots.txt (default: off).")
        p.add_argument("--min-text", type=int, default=0, metavar="N",
                       help="Skip pages with fewer than N extracted characters.")
        p.add_argument("--out", default=None, metavar="FILE",
                       help="Save results to file (format inferred from extension).")
        p.add_argument("--fmt", choices=["json", "text", "csv", "markdown", "sqlite"],
                       default="json", help="Output format (default: json).")
        p.add_argument("--no-progress", action="store_true",
                       help="Disable progress output.")
        p.add_argument("--model", default="all-MiniLM-L6-v2", metavar="NAME",
                       help="Sentence-transformers model for semantic mode.")

    # ── crawl subcommand ──────────────────────────────────────────────────────
    crawl_p = sub.add_parser("crawl", help="Crawl seed URLs and rank by relevance.")
    _add_common(crawl_p)

    # ── deep subcommand ───────────────────────────────────────────────────────
    deep_p = sub.add_parser("deep", help="Deep crawl: follow links + pagination + dedup.")
    _add_common(deep_p)
    deep_p.add_argument("--depth", type=int, default=2, metavar="N",
                        help="Link-follow depth (default: 2).")
    deep_p.add_argument("--max-pages", type=int, default=100, metavar="N",
                        help="Max total pages to crawl (default: 100).")
    deep_p.add_argument("--max-per-domain", type=int, default=50, metavar="N",
                        help="Max pages per domain (default: 50).")
    deep_p.add_argument("--follow-external", action="store_true",
                        help="Follow links to external domains.")
    deep_p.add_argument("--no-paginate", action="store_true",
                        help="Disable automatic pagination following.")
    deep_p.add_argument("--max-pagination", type=int, default=5, metavar="N",
                        help="Max pagination pages per seed URL (default: 5).")
    deep_p.add_argument("--similarity", type=float, default=0.85, metavar="T",
                        help="Content dedup similarity threshold 0-1 (default: 0.85).")
    deep_p.add_argument("--use-mmap", action="store_true",
                        help="Use memory-mapped storage for crawl results.")
    deep_p.add_argument("--domain-rps", action="append", metavar="DOMAIN:RPS",
                        help="Per-domain rate limit e.g. --domain-rps example.com:2.0")

    # ── search subcommand ─────────────────────────────────────────────────────
    search_p = sub.add_parser("search", help="Search DuckDuckGo, crawl results, rank by relevance.")
    search_p.add_argument("query", help="Search query.")
    search_p.add_argument("-n", "--max-results", type=int, default=10, metavar="N",
                          help="Max DDG results to fetch (default: 10).")
    search_p.add_argument("--news", action="store_true",
                          help="Search news instead of web.")
    search_p.add_argument("--timelimit", default=None, choices=["d", "w", "m", "y"],
                          help="Time filter: d=day, w=week, m=month, y=year.")
    search_p.add_argument("--region", default="wt-wt", metavar="CODE",
                          help="DDG region code (default: wt-wt = worldwide).")
    search_p.add_argument("-k", "--top-k", type=int, default=None, metavar="N")
    search_p.add_argument("--mode", choices=["keyword", "semantic", "both"], default="both")
    search_p.add_argument("--semantic-weight", type=float, default=0.5)
    search_p.add_argument("--timeout", type=int, default=10)
    search_p.add_argument("--connections", type=int, default=20)
    search_p.add_argument("--retries", type=int, default=3)
    search_p.add_argument("--rate-limit", type=float, default=None)
    search_p.add_argument("--proxy", action="append", dest="proxies", metavar="URL")
    search_p.add_argument("--cache", action="store_true")
    search_p.add_argument("--cache-path", default=".crl_cache")
    search_p.add_argument("--cache-ttl", type=int, default=86400)
    search_p.add_argument("--robots", action="store_true")
    search_p.add_argument("--min-text", type=int, default=0)
    search_p.add_argument("--out", default=None, metavar="FILE")
    search_p.add_argument("--fmt", choices=["json", "text", "csv", "markdown", "sqlite"],
                          default="json")
    search_p.add_argument("--no-progress", action="store_true")
    search_p.add_argument("--model", default="all-MiniLM-L6-v2")

    # ── sitemap subcommand ────────────────────────────────────────────────────
    sitemap_p = sub.add_parser("sitemap", help="Fetch all URLs from a site's sitemap.xml.")
    sitemap_p.add_argument("url", help="Any URL on the target site.")
    sitemap_p.add_argument("-n", "--max-urls", type=int, default=10000, metavar="N",
                           help="Max URLs to return (default: 10000).")
    sitemap_p.add_argument("--timeout", type=int, default=10)
    sitemap_p.add_argument("--out", default=None, metavar="FILE",
                           help="Save URL list to file (one per line).")

    # ── js subcommand ─────────────────────────────────────────────────────────
    js_p = sub.add_parser("js", help="JS-render URLs with Playwright then rank by relevance.")
    _add_common(js_p)
    js_p.add_argument("--wait", default="networkidle",
                      choices=["networkidle", "load", "domcontentloaded"],
                      help="When to consider page loaded (default: networkidle).")
    js_p.add_argument("--wait-for", default=None, metavar="SELECTOR",
                      help="Wait for CSS selector to appear before extracting HTML.")
    js_p.add_argument("--extra-wait", type=int, default=0, metavar="MS",
                      help="Extra ms to wait after page load (default: 0).")
    js_p.add_argument("--max-concurrent", type=int, default=3, metavar="N",
                      help="Max parallel browser pages (default: 3).")
    js_p.add_argument("--no-block", action="store_true",
                      help="Don't block images/fonts/media (slower but more complete).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from .cache import TieredCache
    from .progress import ProgressReporter
    from .output import save, to_json, to_text, to_csv, to_markdown

    # ── sitemap — no cache/progress args ─────────────────────────────────────
    if args.command == "sitemap":
        from .sitemap import fetch_sitemap_urls_sync
        urls = fetch_sitemap_urls_sync(args.url, max_urls=args.max_urls, timeout=args.timeout)
        if not urls:
            print(f"[CRL] No sitemap URLs found for {args.url}", file=sys.stderr)
            print("[CRL] Site may not have a sitemap.xml, or it returned 404.", file=sys.stderr)
            return 1
        if args.out:
            with open(args.out, "w") as f:
                f.write("\n".join(urls))
            print(f"[CRL] Saved {len(urls)} URLs -> {args.out}", file=sys.stderr)
        else:
            print(f"[CRL] Found {len(urls)} URLs", file=sys.stderr)
            print("\n".join(urls))
        return 0

    cache = TieredCache(disk_path=args.cache_path, disk_ttl=args.cache_ttl) if args.cache else None
    progress = None if args.no_progress else ProgressReporter()

    try:
        if args.command == "crawl":
            from . import crawl
            results = crawl(
                urls=args.urls, query=args.query, top_k=args.top_k,
                mode=args.mode, semantic_weight=args.semantic_weight,
                timeout=args.timeout, max_connections=args.connections,
                retries=args.retries, rate_limit=args.rate_limit,
                proxies=args.proxies, cache=cache, respect_robots=args.robots,
                min_text_length=args.min_text, model_name=args.model, progress=progress,
            )

        elif args.command == "search":
            from .search import search_urls, search_news_urls
            from . import crawl as _crawl
            if args.news:
                urls = search_news_urls(args.query, max_results=args.max_results,
                                        region=args.region, timelimit=args.timelimit)
            else:
                urls = search_urls(args.query, max_results=args.max_results,
                                   region=args.region, timelimit=args.timelimit)
            if not urls:
                print("[CRL] No URLs found.", file=sys.stderr)
                return 1
            print(f"[CRL] Found {len(urls)} URLs from DDG.", file=sys.stderr)
            results = _crawl(
                urls=urls, query=args.query, top_k=args.top_k,
                mode=args.mode, semantic_weight=args.semantic_weight,
                timeout=args.timeout, max_connections=args.connections,
                retries=args.retries, rate_limit=args.rate_limit,
                proxies=args.proxies, cache=cache, respect_robots=args.robots,
                min_text_length=args.min_text, model_name=args.model, progress=progress,
            )

        elif args.command == "js":
            from .js_renderer import js_crawl_sync, is_available
            if not is_available():
                print("[CRL] Playwright not installed.", file=sys.stderr)
                print("[CRL] Run: pip install playwright && playwright install chromium",
                      file=sys.stderr)
                return 1
            results = js_crawl_sync(
                urls=args.urls, query=args.query, top_k=args.top_k,
                mode=args.mode, semantic_weight=args.semantic_weight,
                wait_until=args.wait, wait_for_selector=args.wait_for,
                extra_wait_ms=args.extra_wait, max_concurrent=args.max_concurrent,
                block_resources=not args.no_block,
                proxy=args.proxies[0] if args.proxies else None,
                min_text_length=args.min_text, model_name=args.model,
            )

        else:  # deep
            from . import deep_search
            domain_rps = {}
            for entry in (args.domain_rps or []):
                try:
                    d, r = entry.rsplit(":", 1)
                    domain_rps[d.strip()] = float(r.strip())
                except ValueError:
                    print(f"[CRL] Invalid --domain-rps format: {entry}", file=sys.stderr)
                    return 1
            results = deep_search(
                urls=args.urls, query=args.query, depth=args.depth,
                max_pages=args.max_pages, max_pages_per_domain=args.max_per_domain,
                follow_external=args.follow_external, paginate=not args.no_paginate,
                max_pagination_pages=args.max_pagination, top_k=args.top_k,
                mode=args.mode, semantic_weight=args.semantic_weight,
                timeout=args.timeout, max_connections=args.connections,
                retries=args.retries, rate_limit=args.rate_limit,
                domain_rate_limits=domain_rps or None, proxies=args.proxies,
                cache=cache, respect_robots=args.robots, min_text_length=args.min_text,
                similarity_threshold=args.similarity, model_name=args.model,
                progress=progress, use_mmap=args.use_mmap,
            )

    except ValueError as exc:
        print(f"[CRL] Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[CRL] Interrupted.", file=sys.stderr)
        return 130

    # ── Output ────────────────────────────────────────────────────────────────
    if args.out:
        ext = args.out.rsplit(".", 1)[-1].lower()
        ext_map = {"json": "json", "csv": "csv", "txt": "text",
                   "md": "markdown", "db": "sqlite", "sqlite": "sqlite"}
        fmt = ext_map.get(ext, args.fmt)
        save(results, args.out, fmt=fmt)
        print(f"[CRL] Saved {len(results)} results -> {args.out}", file=sys.stderr)
    else:
        formatters = {
            "json": to_json, "text": to_text,
            "csv": to_csv, "markdown": to_markdown,
        }
        fmt = args.fmt if args.fmt in formatters else "json"
        if args.fmt == "sqlite":
            print("[CRL] sqlite format requires --out <file.db>", file=sys.stderr)
            return 1
        print(formatters[fmt](results))

    return 0


def entry_point() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry_point()

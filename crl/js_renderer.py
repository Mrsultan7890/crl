"""
JS Renderer — Playwright-based JavaScript rendering for dynamic sites.

Handles React, Next.js, Vue, Angular, and any JS-heavy site that returns
empty HTML without browser execution.

Pure Python API. Requires:
    pip install playwright
    playwright install chromium

Falls back gracefully to plain httpx fetch if Playwright not installed.
"""

import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default wait strategy — wait until network is idle (no requests for 500ms)
_DEFAULT_WAIT = "networkidle"
_DEFAULT_TIMEOUT = 30000  # ms


def is_available() -> bool:
    """Check if Playwright is installed and usable."""
    try:
        from playwright.async_api import async_playwright  # noqa
        return True
    except ImportError:
        return False


async def render_one(
    url: str,
    wait_until: str = _DEFAULT_WAIT,
    timeout: int = _DEFAULT_TIMEOUT,
    headless: bool = True,
    wait_for_selector: Optional[str] = None,
    extra_wait_ms: int = 0,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    block_resources: bool = True,
) -> Dict:
    """
    Render a single URL using Playwright (headless Chromium).

    Args:
        url: URL to render.
        wait_until: When to consider page loaded:
                    'networkidle' (default), 'load', 'domcontentloaded', 'commit'.
        timeout: Max wait time in milliseconds (default 30000).
        headless: Run browser headlessly (default True).
        wait_for_selector: Optional CSS selector to wait for before extracting HTML.
        extra_wait_ms: Extra ms to wait after page load (for lazy JS).
        proxy: Proxy URL e.g. 'http://user:pass@host:port'.
        user_agent: Custom User-Agent string.
        block_resources: Block images/fonts/media to speed up rendering.

    Returns:
        Dict with keys: url, html, status, error (same shape as fetcher.fetch_one).
    """
    try:
        from playwright.async_api import async_playwright, Error as PlaywrightError
    except ImportError:
        return {
            "url": url, "original_url": url, "status": None,
            "html": None, "headers": {}, "error": "playwright not installed",
        }

    proxy_settings = {"server": proxy} if proxy else None

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=headless, proxy=proxy_settings)
            context_kwargs = {}
            if user_agent:
                context_kwargs["user_agent"] = user_agent
            context = await browser.new_context(**context_kwargs)

            if block_resources:
                async def _block(route, request):
                    if request.resource_type in ("image", "media", "font", "stylesheet"):
                        await route.abort()
                    else:
                        await route.continue_()
                await context.route("**/*", _block)

            page = await context.new_page()
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)

            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout)
                except Exception:
                    logger.debug("Selector '%s' not found on %s", wait_for_selector, url)

            if extra_wait_ms > 0:
                await page.wait_for_timeout(extra_wait_ms)

            html = await page.content()
            final_url = page.url
            status = response.status if response else None
            headers = dict(response.headers) if response else {}

            await browser.close()

            return {
                "url": final_url,
                "original_url": url,
                "status": status,
                "html": html,
                "headers": headers,
                "error": None,
                "js_rendered": True,
            }

    except Exception as exc:
        logger.error("JS render failed for %s: %s", url, exc)
        return {
            "url": url, "original_url": url, "status": None,
            "html": None, "headers": {}, "error": str(exc),
        }


async def render_all(
    urls: List[str],
    wait_until: str = _DEFAULT_WAIT,
    timeout: int = _DEFAULT_TIMEOUT,
    headless: bool = True,
    wait_for_selector: Optional[str] = None,
    extra_wait_ms: int = 0,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    block_resources: bool = True,
    max_concurrent: int = 3,
) -> List[Dict]:
    """
    Render multiple URLs concurrently using Playwright.

    Args:
        urls: List of URLs to render.
        max_concurrent: Max parallel browser pages (default 3 — browsers are heavy).
        (other args same as render_one)

    Returns:
        List of dicts in same order as input urls.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _render(url: str) -> Dict:
        async with semaphore:
            return await render_one(
                url, wait_until=wait_until, timeout=timeout,
                headless=headless, wait_for_selector=wait_for_selector,
                extra_wait_ms=extra_wait_ms, proxy=proxy,
                user_agent=user_agent, block_resources=block_resources,
            )

    return list(await asyncio.gather(*[_render(u) for u in urls]))


def render_sync(
    urls: List[str],
    wait_until: str = _DEFAULT_WAIT,
    timeout: int = _DEFAULT_TIMEOUT,
    headless: bool = True,
    wait_for_selector: Optional[str] = None,
    extra_wait_ms: int = 0,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    block_resources: bool = True,
    max_concurrent: int = 3,
) -> List[Dict]:
    """
    Sync wrapper for render_all(). Safe in both sync and async contexts.

    Example::

        from crl.js_renderer import render_sync
        from crl import crawl

        # Render JS-heavy pages first, then crawl+rank
        pages = render_sync(["https://react-app.com"])
        # pages have 'html' key — pass directly to parser
    """
    coro = render_all(
        urls, wait_until=wait_until, timeout=timeout,
        headless=headless, wait_for_selector=wait_for_selector,
        extra_wait_ms=extra_wait_ms, proxy=proxy,
        user_agent=user_agent, block_resources=block_resources,
        max_concurrent=max_concurrent,
    )
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


async def js_crawl(
    urls: List[str],
    query: str,
    top_k: Optional[int] = None,
    mode: str = "both",
    semantic_weight: float = 0.5,
    wait_until: str = _DEFAULT_WAIT,
    timeout: int = _DEFAULT_TIMEOUT,
    wait_for_selector: Optional[str] = None,
    extra_wait_ms: int = 0,
    proxy: Optional[str] = None,
    block_resources: bool = True,
    max_concurrent: int = 3,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """
    JS-render URLs then rank by relevance. One-liner for JS-heavy sites.

    Example::

        import asyncio
        from crl.js_renderer import js_crawl

        results = asyncio.run(js_crawl(
            urls=["https://react-app.com"],
            query="python async",
        ))

    Args:
        urls: URLs to JS-render and rank.
        query: Relevance query.
        top_k: Return top K results.
        mode: 'keyword', 'semantic', or 'both'.
        (other args same as render_all)

    Returns:
        List of ranked page dicts with relevance_score.
    """
    from .parser import parse_many
    from .relevance import rank

    raw_pages = await render_all(
        urls, wait_until=wait_until, timeout=timeout,
        wait_for_selector=wait_for_selector, extra_wait_ms=extra_wait_ms,
        proxy=proxy, block_resources=block_resources,
        max_concurrent=max_concurrent,
    )
    parsed = parse_many(raw_pages, min_text_length=min_text_length)
    return rank(parsed, query, top_k=top_k, mode=mode,
                semantic_weight=semantic_weight, model_name=model_name)


def js_crawl_sync(
    urls: List[str],
    query: str,
    top_k: Optional[int] = None,
    mode: str = "both",
    semantic_weight: float = 0.5,
    wait_until: str = _DEFAULT_WAIT,
    timeout: int = _DEFAULT_TIMEOUT,
    wait_for_selector: Optional[str] = None,
    extra_wait_ms: int = 0,
    proxy: Optional[str] = None,
    block_resources: bool = True,
    max_concurrent: int = 3,
    min_text_length: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """Sync wrapper for js_crawl()."""
    coro = js_crawl(
        urls=urls, query=query, top_k=top_k, mode=mode,
        semantic_weight=semantic_weight, wait_until=wait_until,
        timeout=timeout, wait_for_selector=wait_for_selector,
        extra_wait_ms=extra_wait_ms, proxy=proxy,
        block_resources=block_resources, max_concurrent=max_concurrent,
        min_text_length=min_text_length, model_name=model_name,
    )
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

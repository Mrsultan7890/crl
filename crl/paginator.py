"""
Paginator — pure Python pagination detection and next-page URL extraction.

Detects:
  - Numbered pagination  (?page=2, ?p=3, /page/2/)
  - "Next" link patterns (<a>Next</a>, <a aria-label="next">)
  - Infinite scroll hints (rel="next")
"""

import re
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

_NEXT_TEXT = re.compile(
    r"^\s*(next|next\s*page|›|»|→|more|load\s*more)\s*$",
    re.IGNORECASE,
)
_PAGE_PARAM = re.compile(r"^(page|p|pg|paged|pagenum|start|offset)$", re.IGNORECASE)
_PAGE_PATH = re.compile(r"(/page?/?)(\d+)(/?)$", re.IGNORECASE)


def next_page_url(current_url: str, html_links: List[str], current_page: int) -> Optional[str]:
    """
    Given the current URL and all links found on the page, return the next
    page URL or None if no pagination detected.

    Strategy (in priority order):
      1. rel="next" or aria-label="next" link (passed via html_links with marker)
      2. Explicit next-text anchor matching _NEXT_TEXT
      3. URL query param increment (?page=N → ?page=N+1)
      4. URL path increment (/page/N/ → /page/N+1/)
    """
    # Strategy 1 & 2: look for next-page links in html_links
    for link in html_links:
        if _is_next_link(link, current_url):
            return link

    # Strategy 3: query param increment
    incremented = _increment_query_param(current_url, current_page)
    if incremented:
        return incremented

    # Strategy 4: path increment
    incremented = _increment_path(current_url, current_page)
    if incremented:
        return incremented

    return None


def detect_pagination_links(links: List[str], base_url: str) -> List[str]:
    """
    From a list of absolute links, return only those that look like
    pagination links (next pages of the same domain).
    """
    base_domain = _domain(base_url)
    pagination = []
    for link in links:
        if _domain(link) != base_domain:
            continue
        if _PAGE_PATH.search(link):
            pagination.append(link)
            continue
        parsed = urlparse(link)
        params = parse_qs(parsed.query)
        if any(_PAGE_PARAM.match(k) for k in params):
            pagination.append(link)
    return pagination


def build_page_urls(base_url: str, max_pages: int) -> List[str]:
    """
    Pre-build a list of paginated URLs from a base URL up to max_pages.
    Works for both query-param and path-based pagination.

    Example:
        build_page_urls("https://example.com/results?page=1", 5)
        → ["https://example.com/results?page=2", ..., "?page=5"]
    """
    urls = []
    current = base_url
    for page_num in range(2, max_pages + 1):
        nxt = _increment_query_param(current, page_num - 1) or _increment_path(current, page_num - 1)
        if not nxt or nxt in urls:
            break
        urls.append(nxt)
        current = nxt
    return urls


# ── Private helpers ───────────────────────────────────────────────────────────

def _is_next_link(link: str, current_url: str) -> bool:
    """Heuristic: link text markers injected by parser as fragment."""
    fragment = urlparse(link).fragment
    return bool(_NEXT_TEXT.match(fragment)) if fragment else False


def _increment_query_param(url: str, current_page: int) -> Optional[str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    for key in list(params.keys()):
        if _PAGE_PARAM.match(key):
            try:
                val = int(params[key][0])
                if val == current_page:
                    params[key] = [str(current_page + 1)]
                    new_query = urlencode({k: v[0] for k, v in params.items()})
                    return urlunparse(parsed._replace(query=new_query))
            except (ValueError, IndexError):
                continue
    return None


def _increment_path(url: str, current_page: int) -> Optional[str]:
    parsed = urlparse(url)
    match = _PAGE_PATH.search(parsed.path)
    if match:
        try:
            page_num = int(match.group(2))
            if page_num == current_page:
                new_path = _PAGE_PATH.sub(
                    lambda m: f"{m.group(1)}{current_page + 1}{m.group(3)}",
                    parsed.path,
                )
                return urlunparse(parsed._replace(path=new_path))
        except ValueError:
            pass
    return None


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()

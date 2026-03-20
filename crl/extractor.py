"""
Structured data extractor — pulls semantic data from parsed HTML.

Extracts:
  - Open Graph tags  (<meta property="og:*">)
  - Twitter Card     (<meta name="twitter:*">)
  - JSON-LD          (<script type="application/ld+json">)
  - Tables           (<table> → list of dicts)
  - Lists            (<ul>/<ol> → list of strings)
  - Canonical URL    (<link rel="canonical">)
  - Favicon          (<link rel="icon">)

Pure Python — only beautifulsoup4 + stdlib json.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract(html: str, url: str = "") -> Dict[str, Any]:
    """
    Extract all structured data from an HTML page.

    Args:
        html: Raw HTML string.
        url: Source URL (for context only).

    Returns:
        Dict with keys:
          - open_graph: dict of og:* values
          - twitter_card: dict of twitter:* values
          - json_ld: list of parsed JSON-LD objects
          - tables: list of tables (each table = list of row dicts)
          - lists: list of lists (each list = list of strings)
          - canonical: canonical URL string or None
          - favicon: favicon URL string or None
    """
    if not html:
        return _empty()

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        logger.warning("extractor: failed to parse %s: %s", url, exc)
        return _empty()

    return {
        "open_graph": _extract_open_graph(soup),
        "twitter_card": _extract_twitter_card(soup),
        "json_ld": _extract_json_ld(soup),
        "tables": _extract_tables(soup),
        "lists": _extract_lists(soup),
        "canonical": _extract_canonical(soup),
        "favicon": _extract_favicon(soup),
    }


def extract_and_merge(page: Dict) -> Dict:
    """
    Extract structured data from a parsed page dict and merge into it.

    Takes a page dict (from parser.parse()) and adds a 'structured' key.

    Args:
        page: Dict with at least 'html' or existing page fields.

    Returns:
        Same dict with 'structured' key added.
    """
    html = page.get("html") or page.get("text") or ""
    url = page.get("url", "")
    page["structured"] = extract(html, url)
    return page


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty() -> Dict:
    return {
        "open_graph": {},
        "twitter_card": {},
        "json_ld": [],
        "tables": [],
        "lists": [],
        "canonical": None,
        "favicon": None,
    }


def _extract_open_graph(soup: BeautifulSoup) -> Dict[str, str]:
    og = {}
    for tag in soup.find_all("meta", property=True):
        prop = tag.get("property", "")
        if prop.startswith("og:"):
            key = prop[3:]  # strip "og:"
            content = tag.get("content", "").strip()
            if key and content:
                og[key] = content
    return og


def _extract_twitter_card(soup: BeautifulSoup) -> Dict[str, str]:
    tc = {}
    for tag in soup.find_all("meta", attrs={"name": True}):
        name = tag.get("name", "")
        if name.startswith("twitter:"):
            key = name[8:]  # strip "twitter:"
            content = tag.get("content", "").strip()
            if key and content:
                tc[key] = content
    return tc


def _extract_json_ld(soup: BeautifulSoup) -> List[Any]:
    results = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string
        if not raw:
            continue
        raw = raw.strip()
        try:
            data = json.loads(raw)
            # JSON-LD can be a single object or an array
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except json.JSONDecodeError as exc:
            logger.debug("JSON-LD parse error: %s", exc)
    return results


def _extract_tables(soup: BeautifulSoup) -> List[List[Dict[str, str]]]:
    tables = []
    for table in soup.find_all("table"):
        rows = []
        headers: List[str] = []

        # Try to get headers from <thead> or first <tr> with <th>
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

        if not headers:
            first_row = table.find("tr")
            if first_row and first_row.find("th"):
                headers = [th.get_text(strip=True) for th in first_row.find_all("th")]

        # Extract data rows from <tbody> or all <tr>
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells or all(c == "" for c in cells):
                continue
            if headers and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
            else:
                # No headers or mismatch — use col_0, col_1, ...
                rows.append({f"col_{i}": v for i, v in enumerate(cells)})

        if rows:
            tables.append(rows)

    return tables


def _extract_lists(soup: BeautifulSoup) -> List[List[str]]:
    # Skip nav/footer lists (usually menus)
    _skip_parents = {"nav", "footer", "header", "aside"}
    results = []
    for lst in soup.find_all(["ul", "ol"]):
        # Skip if inside noise elements
        if any(p.name in _skip_parents for p in lst.parents):
            continue
        items = [li.get_text(strip=True) for li in lst.find_all("li", recursive=False)]
        items = [i for i in items if i]
        if len(items) >= 2:  # skip single-item lists
            results.append(items)
    return results


def _extract_canonical(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find("link", rel="canonical")
    if tag and tag.get("href"):
        return tag["href"].strip()
    return None


def _extract_favicon(soup: BeautifulSoup) -> Optional[str]:
    for rel in ("icon", "shortcut icon", "apple-touch-icon"):
        tag = soup.find("link", rel=rel)
        if tag and tag.get("href"):
            return tag["href"].strip()
    return None

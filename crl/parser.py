import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "svg", "img",
]
_MULTI_SPACE = re.compile(r"\s+")


def parse(html: str, url: str = "", min_text_length: int = 0) -> Dict:
    """
    Parse raw HTML into structured data.

    Args:
        html: Raw HTML string.
        url: Source URL (used for resolving relative links).
        min_text_length: Skip pages whose extracted text is shorter than this.

    Returns:
        Dict with keys: url, title, text, links, meta, language.
    """
    if not html:
        return _empty(url)

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        logger.warning("Failed to parse HTML from %s: %s", url, exc)
        return _empty(url)

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = _extract_title(soup)
    text = _extract_text(soup)
    language = _extract_language(soup)

    if min_text_length and len(text) < min_text_length:
        logger.debug("Skipping %s — text too short (%d chars)", url, len(text))
        return _empty(url)

    links = _extract_links(soup, url)
    meta = _extract_meta(soup)

    from .extractor import extract as _extract
    structured = _extract(html, url)

    return {
        "url": url,
        "title": title,
        "text": text,
        "links": links,
        "meta": meta,
        "language": language,
        "structured": structured,
    }


def parse_many(pages: List[Dict], min_text_length: int = 0) -> List[Dict]:
    """Parse all successfully fetched pages, skipping errored ones."""
    results = []
    for page in pages:
        if page.get("error"):
            logger.debug("Skipping errored page: %s — %s", page.get("url"), page.get("error"))
            continue
        parsed = parse(page["html"], page["url"], min_text_length=min_text_length)
        if parsed["text"]:
            results.append(parsed)
    return results


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty(url: str) -> Dict:
    return {"url": url, "title": None, "text": "", "links": [], "meta": {}, "language": None, "structured": {}}


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return None


def _extract_text(soup: BeautifulSoup) -> str:
    # Prefer main content areas if present
    for selector in ("main", "article", '[role="main"]', "#content", ".content"):
        container = soup.select_one(selector)
        if container:
            return _clean_text(container.get_text(separator=" "))
    return _clean_text(soup.get_text(separator=" "))


def _clean_text(raw: str) -> str:
    return _MULTI_SPACE.sub(" ", raw).strip()


def _extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https") and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def _extract_meta(soup: BeautifulSoup) -> Dict[str, str]:
    meta = {}
    for m in soup.find_all("meta"):
        key = m.get("name") or m.get("property") or m.get("http-equiv")
        value = m.get("content", "").strip()
        if key and value:
            meta[key.lower()] = value
    return meta


def _extract_language(soup: BeautifulSoup) -> Optional[str]:
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        return html_tag["lang"].split("-")[0].lower()
    return None

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Literal

logger = logging.getLogger(__name__)

FormatType = Literal["json", "text", "csv", "markdown", "sqlite"]


def to_json(pages: List[Dict], indent: int = 2) -> str:
    """Serialize results to a JSON string."""
    return json.dumps(pages, ensure_ascii=False, indent=indent, default=str)


def to_dict(pages: List[Dict]) -> List[Dict]:
    """Return results as a plain list of dicts (passthrough)."""
    return pages


def to_text(pages: List[Dict], text_preview: int = 300) -> str:
    """Human-readable text summary of results."""
    lines = []
    for i, p in enumerate(pages, 1):
        lines.append(f"[{i}] {p.get('url', 'N/A')}")
        lines.append(f"    Title     : {p.get('title') or 'N/A'}")
        lines.append(f"    Language  : {p.get('language') or 'N/A'}")
        lines.append(f"    Relevance : {p.get('relevance_score', 'N/A')}")
        lines.append(f"    Keyword   : {p.get('keyword_score', 'N/A')}")
        lines.append(f"    Semantic  : {p.get('semantic_score', 'N/A')}")
        preview = (p.get("text") or "")[:text_preview].replace("\n", " ")
        lines.append(f"    Preview   : {preview}{'...' if len(p.get('text','')) > text_preview else ''}")
        lines.append("")
    return "\n".join(lines)


def to_csv(pages: List[Dict]) -> str:
    """Serialize results to CSV string."""
    import io
    if not pages:
        return ""
    fields = ["url", "title", "relevance_score", "keyword_score", "semantic_score", "language"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(pages)
    return buf.getvalue()


def to_markdown(pages: List[Dict], text_preview: int = 500) -> str:
    """
    Serialize results to Markdown string.

    Each result becomes a section with title, URL, scores, and text preview.
    """
    lines = ["# CRL Crawl Results", ""]
    for i, p in enumerate(pages, 1):
        title = p.get("title") or "Untitled"
        url = p.get("url", "")
        score = p.get("relevance_score", 0)
        lang = p.get("language") or "unknown"
        preview = (p.get("text") or "")[:text_preview].replace("\n", " ").strip()

        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append(f"- **URL**: [{url}]({url})")
        lines.append(f"- **Relevance**: `{score}`")
        lines.append(f"- **Language**: `{lang}`")

        kw = p.get("keyword_score")
        sem = p.get("semantic_score")
        if kw is not None:
            lines.append(f"- **Keyword score**: `{kw}`")
        if sem is not None:
            lines.append(f"- **Semantic score**: `{sem}`")

        # Open Graph if present
        og = (p.get("structured") or {}).get("open_graph", {})
        if og.get("description"):
            lines.append(f"- **OG Description**: {og['description']}")
        if og.get("image"):
            lines.append(f"- **OG Image**: {og['image']}")

        lines.append("")
        if preview:
            lines.append(f"> {preview}{'...' if len(p.get('text', '')) > text_preview else ''}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def to_sqlite(pages: List[Dict], db_path: str) -> None:
    """
    Save results to a SQLite database.

    Creates (or appends to) a 'pages' table with columns:
      url, title, text, language, relevance_score, keyword_score,
      semantic_score, depth, page_num, structured (JSON blob)

    Args:
        pages: Ranked page results.
        db_path: Path to .db file (created if not exists).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT NOT NULL,
            title           TEXT,
            text            TEXT,
            language        TEXT,
            relevance_score REAL,
            keyword_score   REAL,
            semantic_score  REAL,
            depth           INTEGER,
            page_num        INTEGER,
            structured      TEXT,
            crawled_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_url ON pages(url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON pages(relevance_score)")

    rows = []
    for p in pages:
        structured = p.get("structured")
        rows.append((
            p.get("url", ""),
            p.get("title"),
            p.get("text", ""),
            p.get("language"),
            p.get("relevance_score"),
            p.get("keyword_score"),
            p.get("semantic_score"),
            p.get("depth"),
            p.get("page_num"),
            json.dumps(structured, default=str) if structured else None,
        ))

    cur.executemany("""
        INSERT INTO pages
          (url, title, text, language, relevance_score, keyword_score,
           semantic_score, depth, page_num, structured)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    logger.info("Saved %d results to SQLite: %s", len(pages), db_path)


def save(pages: List[Dict], filepath: str, fmt: FormatType = "json") -> None:
    """
    Save results to a file.

    Args:
        pages: Ranked page results.
        filepath: Destination file path.
        fmt: Output format — 'json', 'text', 'csv', 'markdown', or 'sqlite'.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "sqlite":
        to_sqlite(pages, filepath)
        return

    writers = {
        "json": lambda: to_json(pages),
        "text": lambda: to_text(pages),
        "csv": lambda: to_csv(pages),
        "markdown": lambda: to_markdown(pages),
    }
    if fmt not in writers:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: json, text, csv, markdown, sqlite.")

    content = writers[fmt]()
    path.write_text(content, encoding="utf-8")
    logger.info("Saved %d results to %s (fmt=%s)", len(pages), filepath, fmt)

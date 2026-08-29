"""SQLite storage for the corpus: one row per image + its provenance metadata."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    local_path TEXT NOT NULL,
    source_url TEXT,
    brand_name TEXT,
    source_type TEXT NOT NULL,      -- 'curated_scrape' | 'hf_dataset'
    style_tag TEXT,                 -- one of the PLAN.md style categories, or a HF dataset category
    width INTEGER,
    height INTEGER,
    scraped_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute(SCHEMA)
    return conn


def upsert_image(
    conn: sqlite3.Connection,
    *,
    id: str,
    local_path: str,
    source_type: str,
    source_url: str | None = None,
    brand_name: str | None = None,
    style_tag: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO images (id, local_path, source_url, brand_name, source_type, style_tag, width, height)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            local_path=excluded.local_path,
            source_url=excluded.source_url,
            brand_name=excluded.brand_name,
            source_type=excluded.source_type,
            style_tag=excluded.style_tag,
            width=excluded.width,
            height=excluded.height
        """,
        (id, local_path, source_url, brand_name, source_type, style_tag, width, height),
    )


if __name__ == "__main__":
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"DB at {DB_PATH}, {n} images so far")
    conn.close()

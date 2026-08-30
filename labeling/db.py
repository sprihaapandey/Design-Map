"""Labels + seed-set tables, sharing the corpus SQLite DB from scraper/db.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper.db import get_conn  # noqa: F401 (re-exported for callers)

SCHEMA = """
CREATE TABLE IF NOT EXISTS seed_images (
    image_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS labels (
    image_id TEXT NOT NULL,
    axis TEXT NOT NULL,
    score INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'hand',   -- 'hand' | 'gemini_auto'
    calibrated_score REAL,                 -- score adjusted to hand-label scale; NULL = use score as-is
    labeled_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (image_id, axis)
);

-- Kept entirely separate from `labels` on purpose: this table exists only to
-- compare Gemini's scores against hand scores on the SAME image+axis, which
-- the single-row-per-(image_id, axis) `labels` table can't represent without
-- a source collision (see incident notes in gemini_calibrate.py).
CREATE TABLE IF NOT EXISTS calibration_scores (
    image_id TEXT NOT NULL,
    axis TEXT NOT NULL,
    score INTEGER NOT NULL,
    scored_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (image_id, axis)
);
"""


def get_labeling_conn():
    conn = get_conn()
    conn.executescript(SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(labels)").fetchall()}
    if "source" not in existing_cols:
        conn.execute("ALTER TABLE labels ADD COLUMN source TEXT NOT NULL DEFAULT 'hand'")
    if "calibrated_score" not in existing_cols:
        conn.execute("ALTER TABLE labels ADD COLUMN calibrated_score REAL")
    return conn


class HandLabelProtectedError(Exception):
    """Raised when something tried to overwrite a hand label with a non-hand source."""


def upsert_label(conn, image_id: str, axis: str, score: int, source: str = "hand") -> None:
    if source != "hand":
        existing = conn.execute(
            "SELECT source FROM labels WHERE image_id = ? AND axis = ?", (image_id, axis)
        ).fetchone()
        if existing and existing[0] == "hand":
            raise HandLabelProtectedError(
                f"refusing to overwrite hand label for {image_id}/{axis} with source={source!r}"
            )
    conn.execute(
        """
        INSERT INTO labels (image_id, axis, score, source)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id, axis) DO UPDATE SET score=excluded.score, source=excluded.source, labeled_at=datetime('now')
        """,
        (image_id, axis, score, source),
    )


def upsert_calibration_score(conn, image_id: str, axis: str, score: int) -> None:
    conn.execute(
        """
        INSERT INTO calibration_scores (image_id, axis, score)
        VALUES (?, ?, ?)
        ON CONFLICT(image_id, axis) DO UPDATE SET score=excluded.score, scored_at=datetime('now')
        """,
        (image_id, axis, score),
    )


def seed_ids(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT image_id FROM seed_images").fetchall()]


def labeled_axis_count(conn, image_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM labels WHERE image_id = ?", (image_id,)).fetchone()[0]

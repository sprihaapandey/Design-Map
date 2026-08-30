"""Phase 3: pick a ~90-image seed set for hand-labeling, stratified so every
curated style bucket is represented (not just whatever the DB happens to
return first) plus a slice of the HF-dataset images for generic-web
diversity beyond the 9 curated categories.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from labeling.db import get_labeling_conn

PER_STYLE = 8  # curated_scrape: 9 styles * 8 = 72
HF_SAMPLE = 18  # + 18 from hf_dataset = ~90 total

random.seed(42)


def main() -> None:
    conn = get_labeling_conn()

    existing = conn.execute("SELECT COUNT(*) FROM seed_images").fetchone()[0]
    if existing:
        print(f"seed set already exists ({existing} images) — not regenerating")
        return

    styles = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT style_tag FROM images WHERE source_type='curated_scrape' AND style_tag IS NOT NULL"
        ).fetchall()
    ]

    picked: list[str] = []
    for style in styles:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM images WHERE source_type='curated_scrape' AND style_tag = ?", (style,)
            ).fetchall()
        ]
        random.shuffle(ids)
        picked.extend(ids[:PER_STYLE])

    hf_ids = [r[0] for r in conn.execute("SELECT id FROM images WHERE source_type='hf_dataset'").fetchall()]
    random.shuffle(hf_ids)
    picked.extend(hf_ids[:HF_SAMPLE])

    conn.executemany("INSERT OR IGNORE INTO seed_images (image_id) VALUES (?)", [(i,) for i in picked])
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM seed_images").fetchone()[0]
    print(f"seed set: {total} images ({len(styles)} styles x up to {PER_STYLE} + {HF_SAMPLE} hf_dataset)")
    conn.close()


if __name__ == "__main__":
    main()

"""Phase 1 QA pass: purge captures that aren't actually usable design images —
bot-challenge pages, dead-domain/parking pages, and exact-duplicate renders
(e.g. the same Google login page under a dozen locale subdomains). Found by
spot-checking the smallest files in the corpus, which reliably correlated
with junk (median real screenshot is >100KB; every file checked under ~30KB
was a blank page, "Access Denied", captcha, or "coming soon" placeholder).
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGES_DIR
from scraper.db import get_conn

MIN_BYTES = 30_000

# Confirmed-bad by manual review, just above the size cutoff.
MANUAL_BLOCKLIST = {
    "hf_buydomains-com",  # Cloudflare "verify you are human" parking page
    "hf_com-com",  # "coming soon" domain-parking page
    "hf_bestfreecams-club",  # adult content redirect, not a design capture
}


def main() -> None:
    conn = get_conn()
    rows = conn.execute("SELECT id, local_path FROM images").fetchall()

    by_hash: dict[str, list[str]] = {}
    for image_id, local_path in rows:
        path = IMAGES_DIR.parent.parent / local_path
        if not path.exists():
            continue
        h = hashlib.md5(path.read_bytes()).hexdigest()
        by_hash.setdefault(h, []).append(image_id)

    to_remove: set[str] = set(MANUAL_BLOCKLIST)

    # too small to be a real rendered page
    for image_id, local_path in rows:
        path = IMAGES_DIR.parent.parent / local_path
        if path.exists() and path.stat().st_size < MIN_BYTES:
            to_remove.add(image_id)

    # exact-duplicate renders: keep the first, drop the rest
    for h, ids in by_hash.items():
        if len(ids) > 1:
            to_remove.update(ids[1:])

    print(f"removing {len(to_remove)} of {len(rows)} images")
    id_to_path = {image_id: local_path for image_id, local_path in rows}
    for image_id in to_remove:
        local_path = id_to_path.get(image_id)
        if local_path:
            path = IMAGES_DIR.parent.parent / local_path
            path.unlink(missing_ok=True)
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"corpus now has {remaining} images")
    conn.close()


if __name__ == "__main__":
    main()

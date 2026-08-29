"""Phase 1: pull a deduped sample of real-website screenshots from an existing
free HF dataset, per PLAN.md ("check HuggingFace datasets first").

Source: YashJain/UI-Elements-Detection-Dataset (Apache 2.0) — real screenshots
of 300+ popular websites, filenames encode category + domain, e.g.
`mostvisited_amazon.co.uk_1729630392.png`. We only want the raw images (not
its YOLO element annotations), deduped to one image per domain for max
stylistic diversity, capped at a sample size.
"""

import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGES_DIR
from scraper.db import get_conn, upsert_image

REPO_ID = "YashJain/UI-Elements-Detection-Dataset"
SAMPLE_SIZE = 450

NAME_RE = re.compile(r"^(.+)_((?:[\w-]+\.)+[a-z]{2,})_(\d+)\.png$")


def parse_filename(filename: str) -> tuple[str, str] | None:
    m = NAME_RE.match(filename)
    if not m:
        return None
    category, domain = m.group(1), m.group(2)
    return category, domain


def main() -> None:
    files = list_repo_files(REPO_ID, repo_type="dataset")
    imgs = [f for f in files if f.endswith(".png") and "/images/" in f]

    by_domain: dict[str, tuple[str, str, str]] = {}
    for f in imgs:
        filename = f.split("/")[-1]
        parsed = parse_filename(filename)
        if not parsed:
            continue
        category, domain = parsed
        if domain not in by_domain:
            by_domain[domain] = (f, category, domain)

    picked = list(by_domain.values())[:SAMPLE_SIZE]
    print(f"{len(imgs)} images / {len(by_domain)} unique domains available, downloading {len(picked)}")

    conn = get_conn()
    downloaded = 0
    for repo_path, category, domain in picked:
        image_id = f"hf_{domain.replace('.', '-')}"
        dest = IMAGES_DIR / f"{image_id}.png"
        if not dest.exists():
            local = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
            dest.write_bytes(Path(local).read_bytes())

        from PIL import Image

        with Image.open(dest) as im:
            width, height = im.size

        upsert_image(
            conn,
            id=image_id,
            local_path=str(dest.relative_to(IMAGES_DIR.parent.parent)),
            source_type="hf_dataset",
            source_url=f"https://{domain}",
            brand_name=domain,
            style_tag=category,
            width=width,
            height=height,
        )
        downloaded += 1
        if downloaded % 25 == 0:
            conn.commit()
            print(f"  {downloaded}/{len(picked)}")

    conn.commit()
    conn.close()
    print(f"done: {downloaded} images ingested from {REPO_ID}")


if __name__ == "__main__":
    main()

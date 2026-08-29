"""Run the scan_overlays.py check against only the ids listed in a file
(one id per line) — used to verify a re-scrape fixed what it targeted
without re-scanning the whole corpus.
"""

import sys
from pathlib import Path

import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, IMAGES_DIR
from scraper.db import get_conn
from scraper.scan_overlays import CLEAN_INDEX, LABELS

IDS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/cookie_flagged.txt")


def main() -> None:
    ids = [line.strip() for line in IDS_FILE.read_text().splitlines() if line.strip()]
    conn = get_conn()
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT id, local_path FROM images WHERE id IN ({placeholders})", ids).fetchall()
    conn.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device).eval()

    with torch.no_grad():
        text_features = model.encode_text(tokenizer(LABELS).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for image_id, local_path in rows:
            path = IMAGES_DIR.parent.parent / local_path
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            best = sims.argmax().item()
            status = "CLEAN" if best == CLEAN_INDEX else f"STILL BAD: {LABELS[best]}"
            margin = sims[best].item() - sims[CLEAN_INDEX].item()
            print(f"  {margin:+.4f}  {image_id}: {status}")


if __name__ == "__main__":
    main()

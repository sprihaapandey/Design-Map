"""QA pass: zero-shot CLIP scan for images where a cookie-consent banner,
CAPTCHA/bot-check, or error page is covering the actual design — the kind of
bad capture that survives the blank-page / duplicate-hash filter in
clean_corpus.py because the page did render, just not the thing we wanted.

Not the real Phase 2 embedding pipeline (no caching, uses a temporary
CLIP load) — a cheap way to point re-scraping/removal at the right images
before we commit to the corpus.
"""

import sys
from pathlib import Path

import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, IMAGES_DIR
from scraper.db import get_conn

LABELS = [
    "a clean webpage homepage screenshot showing the full page design",
    "a cookie consent popup or privacy banner overlaying a webpage",
    "a CAPTCHA or 'verify you are human' bot-check challenge screen",
    "an error page such as 404, access denied, or site not found",
]
CLEAN_INDEX = 0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device).eval()

    with torch.no_grad():
        text_tokens = tokenizer(LABELS).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    conn = get_conn()
    rows = conn.execute("SELECT id, local_path FROM images ORDER BY id").fetchall()

    flagged = []
    with torch.no_grad():
        for i, (image_id, local_path) in enumerate(rows):
            path = IMAGES_DIR.parent.parent / local_path
            try:
                img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            except Exception as e:
                print(f"  unreadable: {image_id}: {e}")
                continue
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            best = sims.argmax().item()
            if best != CLEAN_INDEX:
                margin = sims[best].item() - sims[CLEAN_INDEX].item()
                flagged.append((image_id, LABELS[best], margin))
            if (i + 1) % 100 == 0:
                print(f"  scanned {i + 1}/{len(rows)}")

    conn.close()

    flagged.sort(key=lambda x: -x[2])
    print(f"\n{len(flagged)} of {len(rows)} flagged as non-clean (sorted by confidence margin):")
    for image_id, label, margin in flagged:
        print(f"  {margin:+.4f}  {image_id}: {label}")

    out = Path(__file__).parent / "flagged_overlays.txt"
    out.write_text("\n".join(f"{image_id}\t{margin:.4f}\t{label}" for image_id, label, margin in flagged) + "\n")
    print(f"\nwrote {len(flagged)} ids to {out}")


if __name__ == "__main__":
    main()

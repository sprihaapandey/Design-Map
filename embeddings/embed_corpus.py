"""Phase 2: run the full corpus through the frozen CLIP backbone and cache
the resulting embeddings, keyed by image ID.

Output: data/cache/embeddings.npy (float32, L2-normalized, shape [N, D])
        data/cache/ids.txt (one image id per line, same order as embeddings.npy)
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED, IMAGES_DIR
from scraper.db import get_conn

BATCH_SIZE = 32
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    model = model.to(device).eval()

    conn = get_conn()
    rows = conn.execute("SELECT id, local_path FROM images ORDER BY id").fetchall()
    conn.close()
    print(f"{len(rows)} images to embed")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), BATCH_SIZE)):
            batch = rows[i : i + BATCH_SIZE]
            imgs, ids = [], []
            for image_id, local_path in batch:
                path = IMAGES_DIR.parent.parent / local_path
                try:
                    img = preprocess(Image.open(path).convert("RGB"))
                except Exception as e:
                    print(f"  skipping unreadable {image_id}: {e}")
                    continue
                imgs.append(img)
                ids.append(image_id)

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs).to(device)
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())
            all_ids.extend(ids)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(EMBEDDINGS_PATH, embeddings)
    IDS_PATH.write_text("\n".join(all_ids) + "\n")

    print(f"saved {embeddings.shape} to {EMBEDDINGS_PATH}")
    print(f"saved {len(all_ids)} ids to {IDS_PATH}")


if __name__ == "__main__":
    main()

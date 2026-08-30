"""Phase 4: for each reference brand, average its CLIP embeddings across
every captured page (homepage + pricing/features where available) into one
reference vector, then run the trained probe on that average to get the
brand's axis scores.

Averaging the embedding first and running the probe once (rather than
averaging per-page probe outputs) is equivalent for a linear model
(Ridge is affine: probe(mean(x)) == mean(probe(x))) and cheaper.

Output: data/cache/brand_vectors.joblib —
    {brand_name: {"embedding": np.ndarray[512], "axis_scores": {axis: float},
                  "image_ids": [...], "n_images": int}}
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, CACHE_DIR
from labeling.db import get_labeling_conn

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"
PROBES_PATH = CACHE_DIR / "probes.joblib"
BRAND_VECTORS_PATH = CACHE_DIR / "brand_vectors.joblib"

BRANDS = [
    "Linear", "Arc", "Stripe", "Notion", "Vercel", "Figma", "Framer",
    "Superhuman", "Raycast", "Cron", "Airtable", "Webflow", "Airbnb",
    "Discord", "Spotify", "Slack", "Robinhood",
]


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().splitlines()
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    probes = joblib.load(PROBES_PATH)

    conn = get_labeling_conn()

    brand_vectors = {}
    for brand in BRANDS:
        rows = conn.execute("SELECT id FROM images WHERE brand_name = ?", (brand,)).fetchall()
        image_ids = [r[0] for r in rows if r[0] in id_to_idx]
        if not image_ids:
            print(f"  SKIP {brand}: no embedded images")
            continue

        brand_embeddings = np.stack([embeddings[id_to_idx[i]] for i in image_ids])
        avg_embedding = brand_embeddings.mean(axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        axis_scores = {
            axis: float(np.clip(probes[axis].predict(avg_embedding.reshape(1, -1))[0], 1, 5))
            for axis in AXES
        }

        brand_vectors[brand] = {
            "embedding": avg_embedding,
            "axis_scores": axis_scores,
            "image_ids": image_ids,
            "n_images": len(image_ids),
        }
        scores_str = "  ".join(f"{a}={axis_scores[a]:.1f}" for a in AXES)
        print(f"{brand:<12} ({len(image_ids)} imgs)  {scores_str}")

    conn.close()
    joblib.dump(brand_vectors, BRAND_VECTORS_PATH)
    print(f"\nsaved {len(brand_vectors)} brand vectors to {BRAND_VECTORS_PATH}")


if __name__ == "__main__":
    main()

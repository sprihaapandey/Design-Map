"""Phase 6: precompute a 2D UMAP projection of the corpus embeddings for the
map view. Precomputed (not live) per PLAN.md — UMAP is too slow to run
per-request and the layout should stay stable between visits anyway.

Output: data/cache/umap_coords.npy — shape [N, 2], same row order as
embeddings.npy / ids.txt.
"""

import sys
from pathlib import Path

import numpy as np
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
UMAP_COORDS_PATH = CACHE_DIR / "umap_coords.npy"


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"running UMAP on {embeddings.shape}")

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    # normalize to [0, 1] on each axis for easy layout in the frontend
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    np.save(UMAP_COORDS_PATH, coords.astype(np.float32))
    print(f"saved {coords.shape} to {UMAP_COORDS_PATH}")


if __name__ == "__main__":
    main()

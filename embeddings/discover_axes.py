"""Phase 3 stretch: PCA on the full embedding matrix to see if any of the
top directions of variance correspond to a visually coherent "vibe" that
isn't one of the 6 hand-picked axes. For each of the top N components,
pull the most extreme images at both ends for visual inspection.

Captioning the extremes happens by direct visual inspection in this
session (already a vision-capable model) rather than a separate VLM API
call — same goal as the plan's "caption the extremes with a VLM call",
cheaper and consistent with how the axes were hand-labeled to begin with.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"

N_COMPONENTS = 8
N_EXTREMES = 5


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().splitlines()

    pca = PCA(n_components=N_COMPONENTS)
    projections = pca.fit_transform(embeddings)

    print(f"{len(ids)} images, {embeddings.shape[1]}-dim embeddings")
    print(f"top {N_COMPONENTS} components explain {pca.explained_variance_ratio_.sum():.1%} of variance\n")

    for comp in range(N_COMPONENTS):
        scores = projections[:, comp]
        order = np.argsort(scores)
        low_ids = [ids[i] for i in order[:N_EXTREMES]]
        high_ids = [ids[i] for i in order[-N_EXTREMES:][::-1]]

        print(f"=== PC{comp + 1} ({pca.explained_variance_ratio_[comp]:.1%} of variance) ===")
        print(f"  low:  {low_ids}")
        print(f"  high: {high_ids}")
        print()


if __name__ == "__main__":
    main()

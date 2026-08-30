"""Apply the trained per-axis probes to every embedded image (not just the
labeled ones) — this is the "axis vectors" store from PLAN.md's system
diagram (Linear probes -> axis vectors), used by the query layer for the
axis-distance half of the blend score.

Output: data/cache/axis_vectors.npy — shape [N, len(AXES)], same row order
as embeddings.npy / ids.txt.
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, CACHE_DIR

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
PROBES_PATH = CACHE_DIR / "probes.joblib"
AXIS_VECTORS_PATH = CACHE_DIR / "axis_vectors.npy"


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    probes = joblib.load(PROBES_PATH)

    axis_vectors = np.zeros((embeddings.shape[0], len(AXES)), dtype=np.float32)
    for i, axis in enumerate(AXES):
        preds = probes[axis].predict(embeddings)
        axis_vectors[:, i] = np.clip(preds, 1, 5)

    np.save(AXIS_VECTORS_PATH, axis_vectors)
    print(f"saved {axis_vectors.shape} to {AXIS_VECTORS_PATH}")
    print(f"axis order: {AXES}")
    print(f"per-axis mean: {axis_vectors.mean(axis=0).round(2)}")
    print(f"per-axis std:  {axis_vectors.std(axis=0).round(2)}")


if __name__ == "__main__":
    main()

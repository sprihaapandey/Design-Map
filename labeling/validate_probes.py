"""Phase 3: show predicted vs. hand-labeled scores on the held-out 20 images,
per axis, so probe quality can be checked by eye against the original
judgment (not just the R2/MAE summary in train_probes.py).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, CACHE_DIR
from labeling.db import get_labeling_conn
from labeling.train_probes import RANDOM_STATE, TEST_SIZE

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"
PROBES_PATH = CACHE_DIR / "probes.joblib"


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().splitlines()
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    probes = joblib.load(PROBES_PATH)

    conn = get_labeling_conn()
    rows = conn.execute("SELECT image_id, axis, score, source FROM labels").fetchall()
    conn.close()

    by_image: dict[str, dict[str, int]] = {}
    source_of: dict[str, str] = {}
    for image_id, axis, score, source in rows:
        by_image.setdefault(image_id, {})[axis] = score
        source_of[image_id] = source

    labeled_ids = [i for i in by_image if i in id_to_idx and len(by_image[i]) == len(AXES)]
    hand_ids = [i for i in labeled_ids if source_of[i] == "hand"]
    _, test_ids = train_test_split(hand_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for axis in AXES:
        print(f"\n=== {axis} ===")
        print(f"{'image_id':<32}{'true':>6}{'pred':>8}{'diff':>8}")
        model = probes[axis]
        diffs = []
        for image_id in test_ids:
            true = by_image[image_id][axis]
            pred = model.predict(embeddings[id_to_idx[image_id]].reshape(1, -1))[0]
            diff = pred - true
            diffs.append(abs(diff))
            print(f"{image_id:<32}{true:>6}{pred:>8.2f}{diff:>+8.2f}")
        print(f"  mean abs diff: {np.mean(diffs):.2f}")


if __name__ == "__main__":
    main()

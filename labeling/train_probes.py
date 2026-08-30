"""Phase 3: train one linear probe per axis on frozen CLIP embeddings ->
hand labels (scikit-learn), with a held-out split for validation.

Ridge regression (not classification) since axis scores are an ordinal
1-5 scale, not unordered classes — we want "predicted 3.4 for a true 4"
to be a near-miss, not a wrong-class label the way softmax classes would be.

Output: data/cache/probes.joblib — {axis: sklearn Ridge model}
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, CACHE_DIR
from labeling.db import get_labeling_conn

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"
PROBES_PATH = CACHE_DIR / "probes.joblib"

TEST_SIZE = 20  # held-out images per PLAN.md's "validate on a held-out 20 images"
RANDOM_STATE = 0


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().splitlines()
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}

    conn = get_labeling_conn()
    rows = conn.execute("SELECT image_id, axis, score, source, calibrated_score FROM labels").fetchall()
    conn.close()

    by_image: dict[str, dict[str, float]] = {}
    source_of: dict[str, str] = {}
    for image_id, axis, score, source, calibrated_score in rows:
        value = calibrated_score if calibrated_score is not None else score
        by_image.setdefault(image_id, {})[axis] = value
        source_of[image_id] = source  # all axes for one image share a source

    labeled_ids = [i for i in by_image if i in id_to_idx and len(by_image[i]) == len(AXES)]
    hand_ids = [i for i in labeled_ids if source_of[i] == "hand"]
    print(f"{len(labeled_ids)} fully-labeled images with embeddings ({len(hand_ids)} hand-labeled)")

    X = np.stack([embeddings[id_to_idx[i]] for i in labeled_ids])

    # Held-out set is drawn from hand labels only, so "validate against your
    # own judgment" checks against real judgment, not Gemini's own output.
    hand_train, test_ids = train_test_split(hand_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_ids = hand_train + [i for i in labeled_ids if source_of[i] == "gemini_auto"]
    train_idx = [labeled_ids.index(i) for i in train_ids]
    test_idx = [labeled_ids.index(i) for i in test_ids]

    probes = {}
    print(f"\ntrain={len(train_ids)}  held-out={len(test_ids)}\n")
    print(f"{'axis':<14}{'train R2':>10}{'test R2':>10}{'test MAE':>10}   alpha")

    for axis in AXES:
        y = np.array([by_image[i][axis] for i in labeled_ids], dtype=np.float32)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        alphas = np.logspace(0, 4, 20)
        cv_model = RidgeCV(alphas=alphas, cv=5)
        cv_model.fit(X_train, y_train)
        model = Ridge(alpha=cv_model.alpha_)
        model.fit(X_train, y_train)

        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))

        print(f"{axis:<14}{train_r2:>10.3f}{test_r2:>10.3f}{mae:>10.3f}   alpha={cv_model.alpha_:.1f}")
        probes[axis] = model

    import joblib

    joblib.dump(probes, PROBES_PATH)
    print(f"\nsaved probes to {PROBES_PATH}")


if __name__ == "__main__":
    main()

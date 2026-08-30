"""Phase 3: fit a per-axis linear correction from (gemini_calibration score
-> hand score) on the 89 images scored by both, then apply it to the 452
gemini_auto scores before they're used as probe training data.

Without this, validate_probes.py showed systematic (not random) overshoot
on minimalism/luxury/editorial once the probe trained on Gemini-heavy data —
Gemini's rubric interpretation runs on a different scale than mine on those
axes, and a plain linear fit corrects for a constant shift/slope, not noise.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES
from labeling.db import get_labeling_conn


def main() -> None:
    conn = get_labeling_conn()
    label_rows = conn.execute("SELECT image_id, axis, score FROM labels WHERE source = 'hand'").fetchall()
    cal_rows = conn.execute("SELECT image_id, axis, score FROM calibration_scores").fetchall()

    hand: dict[str, dict[str, int]] = {}
    for image_id, axis, score in label_rows:
        hand.setdefault(image_id, {})[axis] = score

    gemini_cal: dict[str, dict[str, int]] = {}
    for image_id, axis, score in cal_rows:
        gemini_cal.setdefault(image_id, {})[axis] = score

    paired_ids = [i for i in hand if i in gemini_cal]
    print(f"{len(paired_ids)} images with both hand and gemini_calibration scores\n")

    print(f"{'axis':<14}{'slope':>8}{'intercept':>10}{'raw MAE':>10}{'calib MAE':>10}")
    fits = {}
    for axis in AXES:
        hand_scores = np.array([hand[i][axis] for i in paired_ids], dtype=np.float64)
        gemini_scores = np.array([gemini_cal[i][axis] for i in paired_ids], dtype=np.float64)

        slope, intercept = np.polyfit(gemini_scores, hand_scores, deg=1)
        fits[axis] = (slope, intercept)

        raw_mae = np.mean(np.abs(gemini_scores - hand_scores))
        calibrated = slope * gemini_scores + intercept
        calib_mae = np.mean(np.abs(calibrated - hand_scores))

        print(f"{axis:<14}{slope:>8.2f}{intercept:>10.2f}{raw_mae:>10.2f}{calib_mae:>10.2f}")

    # Apply the fit to every gemini_auto score.
    auto_rows = conn.execute("SELECT image_id, axis, score FROM labels WHERE source = 'gemini_auto'").fetchall()
    for image_id, axis, score in auto_rows:
        slope, intercept = fits[axis]
        calibrated = float(np.clip(slope * score + intercept, 1, 5))
        conn.execute(
            "UPDATE labels SET calibrated_score = ? WHERE image_id = ? AND axis = ?",
            (calibrated, image_id, axis),
        )

    conn.commit()
    print(f"\napplied calibration to {len(auto_rows)} gemini_auto label rows")
    conn.close()


if __name__ == "__main__":
    main()

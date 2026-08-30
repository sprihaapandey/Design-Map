"""Phase 3: auto-label the rest of the corpus (everything not in the hand-
labeled seed set) with Gemini vision, using the same rubric applied during
hand-labeling as the scoring reference. Fills in training data for the
linear probes, which is the real fix for the small-n underfitting problem
found in train_probes.py's first pass (69 hand labels for a 512-dim probe).

Requires GEMINI_API_KEY in the environment.

Run with:
    python labeling/gemini_autolabel.py [--limit N]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, IMAGES_DIR
from labeling.db import get_labeling_conn, seed_ids, upsert_label

MODEL = "gemini-3.5-flash-lite"
# Free tier is 15 requests/minute for this model — 4.5s spacing gives a margin
# under that (13.3 rpm) rather than cutting it exactly to the limit.
REQUEST_DELAY_S = 4.5
MAX_RETRIES = 5

RUBRIC = """You are scoring a website homepage screenshot on six design-taste axes.
Score each axis as an INTEGER from 1 to 5, using these anchors:

- minimalism: 1 = cluttered/maximal, 5 = extremely sparse/minimal
- playfulness: 1 = serious/corporate, 5 = very playful/fun/whimsical
- luxury: 1 = mass-market/cheap feel, 5 = high-end/premium feel
- technical: 1 = consumer-oriented, 5 = highly technical/engineering-oriented
- editorial: 1 = pure product/utility page, 5 = magazine-like/journalistic layout
- density: 1 = lots of whitespace, sparse content, 5 = densely packed with content

Respond with ONLY a JSON object, no other text, no markdown fences:
{"minimalism": <int>, "playfulness": <int>, "luxury": <int>, "technical": <int>, "editorial": <int>, "density": <int>}
"""


def parse_response(text: str) -> dict[str, int] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not all(axis in data for axis in AXES):
        return None
    try:
        return {axis: int(round(float(data[axis]))) for axis in AXES}
    except (TypeError, ValueError):
        return None


def main(limit: int | None) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / "data" / ".env")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set (checked env and data/.env)")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key)

    conn = get_labeling_conn()
    seeded = set(seed_ids(conn))
    all_rows = conn.execute("SELECT id, local_path FROM images").fetchall()
    todo = [(image_id, local_path) for image_id, local_path in all_rows if image_id not in seeded]

    already_labeled = set()
    for image_id, _ in todo:
        count = conn.execute("SELECT COUNT(*) FROM labels WHERE image_id = ?", (image_id,)).fetchone()[0]
        if count >= len(AXES):
            already_labeled.add(image_id)
    todo = [(i, p) for i, p in todo if i not in already_labeled]

    if limit:
        todo = todo[:limit]

    print(f"{len(already_labeled)} already auto-labeled, {len(todo)} to go")

    ok, failed = 0, 0
    for i, (image_id, local_path) in enumerate(todo):
        path = IMAGES_DIR.parent.parent / local_path
        image_bytes = path.read_bytes()

        scores = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        RUBRIC,
                        genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ],
                )
                scores = parse_response(response.text)
                if scores is None:
                    print(f"  UNPARSEABLE {image_id}: {response.text[:200]!r}")
                break
            except Exception as e:
                is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    backoff = REQUEST_DELAY_S * (attempt + 2)
                    print(f"  rate limited on {image_id}, backing off {backoff:.0f}s (attempt {attempt + 1})")
                    time.sleep(backoff)
                    continue
                print(f"  ERROR {image_id}: {e}")
                break

        if scores:
            for axis, score in scores.items():
                score = max(1, min(5, score))
                upsert_label(conn, image_id, axis, score, source="gemini_auto")
            ok += 1
        else:
            failed += 1

        if (i + 1) % 25 == 0:
            conn.commit()
            print(f"  {i + 1}/{len(todo)}  (ok={ok} failed={failed})")

        time.sleep(REQUEST_DELAY_S)

    conn.commit()
    conn.close()
    print(f"done: {ok} labeled, {failed} failed (of {len(todo)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.limit)

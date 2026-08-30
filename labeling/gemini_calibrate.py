"""Phase 3 calibration: run Gemini on the same 89 images that were hand-
labeled, so its scores can be compared directly against mine on identical
images. validate_probes.py showed a systematic (not random) overshoot on
minimalism/luxury/editorial once the probe trained on Gemini-heavy data —
that's a calibration gap between Gemini's rubric interpretation and mine,
and it's fixable since we now have paired scores to measure it from.

Batches several images per API call — the free-tier limit is per-request,
not per-image, so this turns 15 req/min into ~15*BATCH_SIZE images/min
instead of one image per rate-limited slot.

Stores results in the dedicated calibration_scores table — NOT the labels
table. An earlier version wrote here with source='gemini_calibration' at the
same (image_id, axis) primary key as hand labels, and its upsert's
ON CONFLICT silently overwrote 45 hand-labeled images with Gemini's scores,
destroying labels that had no other record. Never write calibration
comparisons into the same table/primary-key space as real training labels.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, IMAGES_DIR
from labeling.db import get_labeling_conn, seed_ids, upsert_calibration_score
from labeling.gemini_autolabel import MAX_RETRIES, MODEL

BATCH_SIZE = 8
REQUEST_DELAY_S = 4.5

BATCH_RUBRIC = """You are scoring website homepage screenshots on six design-taste axes.
Score each axis as an INTEGER from 1 to 5, using these anchors:

- minimalism: 1 = cluttered/maximal, 5 = extremely sparse/minimal
- playfulness: 1 = serious/corporate, 5 = very playful/fun/whimsical
- luxury: 1 = mass-market/cheap feel, 5 = high-end/premium feel
- technical: 1 = consumer-oriented, 5 = highly technical/engineering-oriented
- editorial: 1 = pure product/utility page, 5 = magazine-like/journalistic layout
- density: 1 = lots of whitespace, sparse content, 5 = densely packed with content

You will be given {n} images, in order. Respond with ONLY a JSON array of {n}
objects, in the same order as the images, no other text, no markdown fences:
[{{"minimalism": <int>, "playfulness": <int>, "luxury": <int>, "technical": <int>, "editorial": <int>, "density": <int>}}, ...]
"""


def parse_batch_response(text: str, expected_n: int) -> list[dict[str, int]] | None:
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
    if not isinstance(data, list) or len(data) != expected_n:
        return None
    out = []
    for item in data:
        if not all(axis in item for axis in AXES):
            return None
        try:
            out.append({axis: max(1, min(5, int(round(float(item[axis]))))) for axis in AXES})
        except (TypeError, ValueError):
            return None
    return out


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / "data" / ".env")
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(timeout=30_000))

    conn = get_labeling_conn()
    seeded = seed_ids(conn)
    rows = conn.execute(
        f"SELECT id, local_path FROM images WHERE id IN ({','.join('?' for _ in seeded)})", seeded
    ).fetchall()

    already = set(
        r[0]
        for r in conn.execute(
            "SELECT image_id FROM calibration_scores GROUP BY image_id HAVING COUNT(*) >= ?",
            (len(AXES),),
        ).fetchall()
    )
    todo = [(i, p) for i, p in rows if i not in already]
    print(f"{len(already)} already calibration-scored, {len(todo)} to go")

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    ok, failed = 0, 0

    for batch_num, batch in enumerate(batches):
        image_ids = [i for i, _ in batch]
        image_bytes_list = [(IMAGES_DIR.parent.parent / p).read_bytes() for _, p in batch]
        prompt = BATCH_RUBRIC.format(n=len(batch))

        print(f"  starting batch {batch_num + 1}/{len(batches)} ({len(batch)} images)...", flush=True)
        results = None
        for attempt in range(MAX_RETRIES):
            try:
                contents = [prompt] + [
                    genai.types.Part.from_bytes(data=b, mime_type="image/png") for b in image_bytes_list
                ]
                t0 = time.time()
                response = client.models.generate_content(model=MODEL, contents=contents)
                print(f"    call returned in {time.time() - t0:.1f}s", flush=True)
                results = parse_batch_response(response.text, len(batch))
                if results is None:
                    print(f"  UNPARSEABLE batch {batch_num}: {response.text[:200]!r}")
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < MAX_RETRIES - 1:
                    backoff = REQUEST_DELAY_S * (attempt + 2)
                    print(f"  rate limited on batch {batch_num}, backing off {backoff:.0f}s")
                    time.sleep(backoff)
                    continue
                print(f"  ERROR batch {batch_num}: {e}")
                break

        if results:
            for image_id, scores in zip(image_ids, results):
                for axis, score in scores.items():
                    upsert_calibration_score(conn, image_id, axis, score)
            ok += len(batch)
        else:
            failed += len(batch)

        conn.commit()
        print(f"  batch {batch_num + 1}/{len(batches)}  (ok={ok} failed={failed})")
        time.sleep(REQUEST_DELAY_S)

    conn.close()
    print(f"done: {ok} scored, {failed} failed (of {len(todo)})")


if __name__ == "__main__":
    main()

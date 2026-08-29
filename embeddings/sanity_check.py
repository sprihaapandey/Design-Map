"""Phase 2 sanity check: hand-picked pairs (not sampled by style_tag, which
turned out to be too noisy a proxy for visual similarity — see PLAN.md's own
"style categories reflect corpus-sourcing intent, not fine-grained style" note)
of images I've directly inspected in this session, so I can vouch that the
"similar" pairs genuinely look alike and the "dissimilar" ones genuinely don't.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"

SIMILAR_PAIRS = [
    ("scrape_vercel-com", "scrape_framer-com", "dark bento-grid dev-tool marketing pages"),
    ("scrape_notion-so", "scrape_linear-app", "light minimalist SaaS product pages"),
    ("scrape_are-na", "scrape_brutalistwebsites-com", "plain-text, unstyled brutalist pages"),
    ("scrape_duolingo-com", "scrape_oatly-com", "playful, colorful, illustrated brand pages"),
    ("scrape_supabase-com", "scrape_railway-app", "dark bento-grid dev-tool pages"),
]

DISSIMILAR_PAIRS = [
    ("scrape_are-na", "scrape_loewe-com", "plain text vs. vivid fashion editorial photography"),
    ("scrape_duolingo-com", "scrape_apple-com", "colorful illustration vs. minimal product photography"),
    ("scrape_brutalistwebsites-com", "scrape_vercel-com", "unstyled text vs. dark gradient bento grid"),
    ("scrape_oatly-com", "scrape_stripe-com", "playful hand-drawn vs. clean minimal SaaS"),
    ("scrape_loewe-com", "scrape_railway-app", "fashion photography vs. dark dev-tool UI"),
]


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().splitlines()
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}

    def sim(id_a: str, id_b: str) -> float | None:
        if id_a not in id_to_idx or id_b not in id_to_idx:
            print(f"  SKIP (missing from embeddings): {id_a} / {id_b}")
            return None
        a, b = embeddings[id_to_idx[id_a]], embeddings[id_to_idx[id_b]]
        return float(np.dot(a, b))

    print("=== similar pairs (should score high) ===")
    similar_sims = []
    for id_a, id_b, why in SIMILAR_PAIRS:
        s = sim(id_a, id_b)
        if s is not None:
            similar_sims.append(s)
            print(f"  {id_a}  <->  {id_b}   cos_sim={s:.3f}   ({why})")

    print("\n=== dissimilar pairs (should score low) ===")
    dissimilar_sims = []
    for id_a, id_b, why in DISSIMILAR_PAIRS:
        s = sim(id_a, id_b)
        if s is not None:
            dissimilar_sims.append(s)
            print(f"  {id_a}  <->  {id_b}   cos_sim={s:.3f}   ({why})")

    avg_similar = sum(similar_sims) / len(similar_sims)
    avg_dissimilar = sum(dissimilar_sims) / len(dissimilar_sims)
    print(f"\navg similar-pair cos_sim:    {avg_similar:.3f}")
    print(f"avg dissimilar-pair cos_sim: {avg_dissimilar:.3f}")
    print("PASS: similar pairs score higher" if avg_similar > avg_dissimilar else "FAIL: no separation")


if __name__ == "__main__":
    main()

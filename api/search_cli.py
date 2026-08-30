"""Phase 5: quick CLI to test the query engine before Phase 6 builds a UI.

Usage:
    python api/search_cli.py "Linear + Arc + early Stripe"
    python api/search_cli.py "playful and colorful" --alpha 0.3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.query import QueryEngine
from config import AXES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--llm", action="store_true", help="use LLM-based brand extraction instead of regex")
    args = parser.parse_args()

    engine = QueryEngine()

    if args.llm:
        matched, remaining = engine.extract_with_llm(args.query)
    else:
        matched, remaining = engine.parse_brands(args.query)

    print(f'query: "{args.query}"  (alpha={args.alpha})')
    print(f"  matched brands: {matched or '(none)'}")
    print(f"  remaining text: {remaining or '(none)'}")
    print()

    results = engine.search(args.query, alpha=args.alpha, top_k=args.top_k)
    print(f"{'rank':<5}{'image_id':<40}{'score':>8}{'cos_sim':>9}{'axis_sim':>9}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5}{r.image_id:<40}{r.score:>8.3f}{r.cosine_sim:>9.3f}{r.axis_sim:>9.3f}")


if __name__ == "__main__":
    main()

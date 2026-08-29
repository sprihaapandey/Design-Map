"""Shared configuration for TasteMap. Single source of truth for axes, paths, and model choice."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "tastemap.db"

# Frozen vision-language backbone used for all embeddings.
# ViT-B-32 laion2b is a good quality/speed tradeoff for local/Colab CPU+GPU use.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Taste-space axes (Phase 0 decision). Each is a 1-5 scale during hand-labeling.
# These are "vibe" axes learned as linear probes on frozen CLIP embeddings —
# distinct from the style *categories* in docs/PLAN.md section 1 (Minimalism,
# Brutalism, Swiss Style, etc.), which describe corpus sourcing/diversity targets
# rather than the interpretable axes themselves.
AXES = [
    "minimalism",
    "playfulness",
    "luxury",
    "technical",
    "editorial",
    "density",
]

AXIS_SCALE_MIN = 1
AXIS_SCALE_MAX = 5

# TasteMap

A visual search engine for design, organized around an interpretable taste-space embedding. See [docs/PLAN.md](docs/PLAN.md) for the full build plan.

## Repo structure

- `scraper/` — corpus collection (dataset download + Playwright scraping) → Phase 1
- `embeddings/` — CLIP/SigLIP embedding pipeline → Phase 2
- `labeling/` — hand-labeling tools, axis probe training → Phase 3
- `api/` — query layer + FastAPI backend → Phase 5
- `ui/` — search UI → Phase 6
- `data/` — images, SQLite DB, cached embeddings (gitignored, regenerable)
- `notebooks/` — Colab backup-compute notebooks
- `config.py` — shared config: paths, CLIP model choice, taste axes

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For scraping (Phase 1), also install Playwright browsers:

```bash
playwright install chromium
```

## Axes (Phase 0 decision)

Starting taste-space axes, hand-labeled 1-5 and learned as linear probes on frozen CLIP embeddings:
`minimalism, playfulness, luxury, technical, editorial, density` — see `config.py`.

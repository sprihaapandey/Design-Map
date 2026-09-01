# TasteMap

A visual search engine for design, organized around an interpretable
taste-space embedding. Search by brand name ("Linear + Arc + early Stripe"),
free text ("playful hand-drawn illustrations"), or both — results are ranked
by a blend of raw CLIP similarity and distance along six hand-defined taste
axes, and you can nudge those axes live to re-rank, or click any result to
explore from it via real vector arithmetic in embedding space.

**Live**: https://129-153-85-90.nip.io/

See [docs/PLAN.md](docs/PLAN.md) for the original phased build plan and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for a PCA write-up on whether the
embedding space contains taste axes beyond the six chosen here.

## How it works

```
corpus (564 images) → frozen CLIP embeddings (512-dim)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     6 linear probes,              stored per-image,
     trained on 89 hand +          used for brand
     452 VLM-auto labels           reference vectors
     (bias-calibrated)             (17 brands)
              │                           │
              └─────────────┬─────────────┘
                             ▼
                   query layer (api/query.py):
                   brand regex-match + CLIP text encoder
                   → blended cosine-sim + axis-distance ranking
                             │
                             ▼
                   FastAPI backend + vanilla-JS UI
                   (search, live axis sliders, 2D axis/UMAP map,
                   click-to-explore vector nudging)
```

- **Corpus**: 564 images — 195 curated Playwright screenshots (deliberately
  spanning 9 design-movement categories: minimalism, brutalism, Swiss style,
  editorial, hand-drawn, retro, flat, bento, constructivism), 346 from an
  existing HuggingFace real-website-screenshot dataset, 23 brand
  pricing/features pages for the reference table.
- **Axes**: `minimalism, playfulness, luxury, technical, editorial, density`
  — each a linear (Ridge) probe on frozen CLIP embeddings, trained on 89
  hand-labeled images plus 452 Gemini-auto-labeled ones, with a fitted
  per-axis correction for the systematic gap between the two labeling
  sources. Held-out MAE: 0.50–0.85 on a 1–5 scale, all six axes.
- **Brand reference table**: 17 well-known products (Linear, Arc, Stripe,
  Notion, Vercel, Figma, Framer, and others), each an averaged embedding
  across homepage + pricing/features pages.
- **Query layer**: brand names are regex-matched (no LLM call needed for the
  common case); free text goes through CLIP's zero-shot text encoder;
  scores are min-max normalized per-query before blending, correcting for
  CLIP's text↔image "modality gap" so `alpha` behaves consistently
  regardless of query type.

## Repo structure

- `scraper/` — corpus collection: HF dataset ingestion + Playwright scraping,
  with cookie-banner auto-dismiss and a CLIP-based QA pass for
  bot-blocked/broken captures
- `embeddings/` — CLIP embedding pipeline, per-axis corpus scoring, UMAP
  projection, PCA axis-discovery experiment
- `labeling/` — hand-labeling web UI, Gemini auto-labeling + calibration,
  probe training/validation
- `brands/` — brand reference table (multi-page capture + averaging)
- `api/` — query engine (`query.py`) + FastAPI backend (`server.py`) +
  a CLI for testing queries without the UI (`search_cli.py`)
- `ui/` — the search frontend (single-file HTML/CSS/JS, no build step)
- `deploy/` — production deployment: systemd unit, nginx config, and an
  idempotent provisioning script (`setup.sh`) that handles both
  Ubuntu/Debian and Oracle Linux
- `data/` — images, SQLite DB, cached embeddings/probes (gitignored*,
  regenerable via the scripts above — *committed for this project
  specifically so the deploy script can `git clone` a working corpus
  directly, see `deploy/setup.sh`)
- `notebooks/` — Colab backup-compute notebook for embedding on GPU
- `docs/` — the build plan and experiment write-ups
- `config.py` — shared config: paths, CLIP model choice, taste axes

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # full dev environment
playwright install chromium       # only needed for scraper/ scripts
```

Run the app locally:

```bash
uvicorn api.server:app --port 8010
```

then open http://localhost:8010. Or skip the UI and query from the CLI:

```bash
python api/search_cli.py "Linear + Arc + early Stripe"
```

## Deployment

Runs in production on an Oracle Cloud Always Free ARM VM (chosen over
Render/AWS after measuring that CLIP's text encoder needs ~950MB RAM at
peak — more than typical free-tier PaaS limits, but comfortably inside
Oracle's free 12GB allocation). `requirements-server.txt` is a trimmed
dependency list for just `api/server.py`'s runtime path, deliberately
excluding scraping/labeling-only packages (Playwright, pandas,
transformers). See `deploy/setup.sh` for the full provisioning script —
it's idempotent, so re-running it after a `git push` picks up the latest
code and restarts the service.

## Regenerating the corpus/probes from scratch

Each phase's scripts are runnable independently, in order:

```bash
python scraper/fetch_hf_dataset.py       # pull real-website sample from HF
python scraper/scrape_sites.py           # curated Playwright scrape
python scraper/clean_corpus.py           # drop broken/blank/duplicate captures
python scraper/scan_overlays.py          # CLIP-based QA for cookie banners etc.

python embeddings/embed_corpus.py        # CLIP-embed the whole corpus
python embeddings/score_corpus_axes.py   # apply trained probes to every image
python embeddings/compute_umap.py        # 2D projection for the map view

python labeling/select_seed.py           # pick a stratified hand-labeling set
uvicorn labeling.label_app:app --port 8420   # hand-label via browser
python labeling/gemini_calibrate.py      # score the same set with Gemini
python labeling/apply_calibration.py     # fit + apply the bias correction
python labeling/gemini_autolabel.py      # auto-label the rest of the corpus
python labeling/train_probes.py          # train the 6 axis probes
python labeling/validate_probes.py       # check predictions on held-out hand labels

python brands/capture_extra_pages.py     # extra pages per reference brand
python brands/build_brand_table.py       # averaged brand vectors
```

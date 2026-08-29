# TasteMap — Project Planning Document

**A visual search engine for design, organized around an interpretable taste-space embedding.**

---

## 1. Goal & Scope

Build a system that can:
1. Take a query like *"Find designs that feel like Linear + Arc + early Stripe"* and retrieve visually/semantically similar designs from a corpus.
2. Represent every design not just as a raw CLIP embedding, but as a small set of interpretable axes (minimalism, playfulness, luxury, technical, editorial, density...).
3. Let users navigate the resulting "taste space" interactively — sliders, a 2D map, vector-arithmetic style exploration.
4. Source: https://tilda.education/en/web-design-styles
Consider the following axes: Minimalism, Brutalism & Neobrutalism, Constructivism, Swiss Style, Editorial Style, Hand-Drawn Style, Retro, Flat, Bento Style

**Non-goals for v1:** generating new designs, scoring subjective "quality," supporting every design category (start with web/product UI, not print/branding/illustration).

**Budget constraint:** near-$0. Every design decision below defaults to the free/local option, with paid options noted only where they meaningfully change quality.

---

## 2. System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────────────┐
│   Corpus     │───▶│ CLIP/SigLIP  │───▶│  Raw embeddings     │
│ (images)     │    │  (frozen)    │    │  (cached .npy)      │
└─────────────┘    └──────────────┘    └──────────┬──────────┘
                                                    │
                          ┌─────────────────────────┴──────────────┐
                          ▼                                        ▼
                 ┌─────────────────┐                    ┌────────────────────┐
                 │ Linear probes    │                    │ In-memory / pgvector│
                 │ (per axis)       │                    │ similarity search   │
                 │ → axis vectors   │                    └──────────┬─────────┘
                 └────────┬─────────┘                               │
                          │                                         │
                          ▼                                         ▼
                 ┌──────────────────────────────────────────────────────┐
                 │              Query layer (text / brand-name / hybrid) │
                 │  - brand reference table (avg embedding per brand)    │
                 │  - CLIP text encoder for free text                    │
                 │  - blend: cosine sim + axis-vector distance           │
                 └──────────────────────────┬───────────────────────────┘
                                             ▼
                                  ┌────────────────────┐
                                  │   UI: map, sliders, │
                                  │   nearest neighbors │
                                  └────────────────────┘
```

---

## 3. Phased Build Plan

### Phase 0 — Setup (Day 1)
- [ ] Set up repo structure: `/scraper`, `/embeddings`, `/labeling`, `/api`, `/ui`
- [ ] Install `open_clip` or HuggingFace `transformers` (SigLIP) locally
- [ ] Decide axes (start with 5–6): **minimalism, playfulness, luxury, technical, editorial, density**
- [ ] Set up Google Colab notebook as backup compute if local machine is slow

### Phase 1 — Corpus (Days 1–3)
- [ ] Check HuggingFace datasets first: search "Rico dataset," "WebSight," "UI screenshots" — sample a free existing dataset rather than scraping from zero
- [ ] Supplement with targeted scraping (Playwright script) of ~200–300 real product sites/apps — mix of minimal/maximal, playful/corporate to avoid a skewed corpus
- [ ] Add variety from Dribbble's public API (free tier) if more stylistic range is needed
- [ ] Target: 500–800 images minimum for v1 (more later if needed)
- [ ] Store images + metadata (source URL, brand name if known) in a simple SQLite table

### Phase 2 — Base Embeddings (Day 3–4)
- [ ] Run full corpus through CLIP/SigLIP (batched, local or Colab GPU)
- [ ] Cache embeddings to `.npy` / SQLite, keyed by image ID
- [ ] Sanity check: pick 5 known-similar pairs, confirm cosine similarity is high; pick 5 known-dissimilar pairs, confirm it's low

### Phase 3 — Labeling & Axis Learning (Days 4–7)
- [ ] Hand-label a seed set of ~80–100 images yourself, 1–5 scale per axis (spreadsheet)
- [ ] Train a linear/logistic regression probe per axis on frozen CLIP embeddings → hand labels (scikit-learn)
- [ ] Validate: check probe predictions against your own judgment on a held-out 20 images per axis
- [ ] Use free-tier Gemini/Claude vision API to auto-label the remaining corpus using seed set as few-shot reference, to fill in axis scores for images you didn't hand-label
- [ ] *(Stretch)* Run PCA/NMF on the embedding matrix to discover unnamed axes; caption the extremes with a VLM call to see if they're interpretable

### Phase 4 — Brand Reference Table (Day 7)
- [ ] Manually screenshot 15–20 well-known products (Linear, Arc, Stripe, Notion, Vercel, Figma, Framer, etc.)
- [ ] Compute and store their averaged embedding + averaged axis vector

### Phase 5 — Query Layer (Days 8–9)
- [ ] Brand-name resolution: parse query for known brand names → average their reference vectors → nearest-neighbor search
- [ ] Free-text queries: route through CLIP's text encoder (zero-shot) as the default path
- [ ] Blend score: `similarity = α * cosine(embedding) + (1-α) * axis_vector_distance`, tune α by eye
- [ ] Light LLM call (free tier) to extract brand names / adjectives from a natural-language query before resolution

### Phase 6 — UI (Days 9–12)
- [ ] Search bar + results grid (basic retrieval working end-to-end first)
- [ ] Axis sliders that re-rank results live
- [ ] UMAP 2D projection of corpus for the "map" view (precomputed, not live)
- [ ] Click-to-explore: select a design, nudge sliders, re-query with `current_vector + direction * magnitude`
- [ ] Deploy to Vercel/Netlify free tier

### Phase 7 — Polish & Write-up (Days 12–14)
- [ ] Write up the Bradley-Terry / pairwise approach as documented future work (even if not implemented)
- [ ] Record a short demo video/gif
- [ ] Document the discovered-axes experiment if it produced anything interesting

---

## 4. Cost Tracking

| Item | Est. Cost | Notes |
|---|---|---|
| Embeddings (CLIP/SigLIP, local/Colab) | $0 | |
| Corpus (existing dataset + scraping) | $0 | |
| Hand labeling (your time) | $0 | ~2–3 hrs |
| Auto-labeling buffer (API credits) | $10–20 | Optional, free tier first |
| Vector search (in-memory / Supabase free) | $0 | |
| Hosting (Vercel/Netlify free tier) | $0 | |
| **Total** | **$0–20** | |

---

## 5. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Corpus too narrow (all SaaS landing pages) | Deliberately source playful/maximalist/editorial examples, not just minimal tech sites |
| Hand-labeled axes reflect only your taste | Acceptable for v1 — frame it as *a* taste space, not *the* taste space; note as limitation |
| Linear probe underfits nuanced axes | If quality is poor, fall back to a small MLP head — still cheap, still frozen backbone |
| Free-tier API limits hit mid-labeling | Batch requests, cache aggressively, budget $10–20 as paid backstop |
| Query blending (α) feels arbitrary | Make it a UI slider — turns a weakness into a feature ("raw similarity" ↔ "taste similarity") |

---

## 6. Definition of Done (v1)

- A working search bar that accepts either brand names or free text and returns a ranked grid of visually coherent results.
- At least 5 named, working axes with sliders that visibly change results.
- A 2D map view of the corpus that's navigable.
- Total spend under $20.
- One clear example query (e.g. "Linear + Arc + early Stripe") that produces a genuinely good, explainable result set.
# Experiments

## Discovered-axes via PCA (Phase 3 stretch)

**Method**: ran PCA (`embeddings/discover_axes.py`) on all 542 CLIP embeddings
(512-dim), took the top 8 principal components, and for each pulled the 5
images at both extremes for direct visual inspection (in place of a separate
VLM captioning call — this session is already a vision-capable model, so
inspecting the extremes directly is equivalent and avoids extra API cost).

**Headline result**: the top 8 components together explain only **22.5%** of
total variance (max ~6% for PC1 alone). CLIP embeddings of full-page website
screenshots don't decompose into a handful of dominant "vibe" directions —
variance is spread thin across many entangled factors (layout, color
palette, language/locale, content type, information density, brand identity)
at once. This is itself informative: it suggests the 6 hand-picked axes
(minimalism, playfulness, luxury, technical, editorial, density) are doing
real work by imposing *interpretable* structure the raw embedding space
doesn't offer on its own.

**Per-component findings** (from visual inspection of the extremes):

- **PC4 — a genuinely novel, interpretable axis.** Low end: ecommerce
  marketplaces (Walmart, eBay, Newegg, Rakuten — busy colorful product grids,
  banner ads). High end: reference/documentation/academic sites (JSTOR, MDN,
  Hugging Face, Medium, Wikipedia — clean typographic layouts, some dark-mode
  dev docs, some restrained editorial whitespace). This reads as a
  **content-type axis** ("commerce" vs. "reference/knowledge") distinct from
  any of the 6 hand-picked style axes. Candidate for a 7th axis if extended
  in a future pass — not added here since it would require re-labeling the
  seed set.
- **PC1 and PC6 — redundant with existing axes.** Both isolate the same
  cluster: dense international news portals (TMZ, Le Monde, Nikkei, NYT,
  Yahoo Finance) at one extreme vs. cleaner product/editorial pages at the
  other. Substantially overlaps the existing `editorial` and `density` axes
  — not new signal. Notable sanity check along the way: the `scrape_` and
  `hf_` captures of nytimes.com (from two different corpus sources) landed
  next to each other in this component, confirming the embedding space is
  picking up genuine visual similarity rather than noise.
- **PC2, PC3, PC5, PC7, PC8** — no describable pattern on inspection; likely
  residual noise, consistent with each explaining well under 3% of variance.

**Conclusion**: the 6 chosen axes already cover the main interpretable
structure in the corpus. One solid candidate (commerce vs. reference) exists
for a possible 7th axis but wasn't pursued further for v1.

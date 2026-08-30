"""Phase 5: query layer. Resolves a natural-language query (brand names,
free text, or a mix — e.g. "Linear + Arc + early Stripe" or "playful and
colorful" or "minimalist dark mode dev tools like Vercel") to a ranked list
of corpus images.

Brand-name resolution: PLAN.md calls for parsing brand names out of the
query with "a light LLM call." Regex word-matching against the 17 known
reference brands handles the stated example query ("Linear + Arc + early
Stripe") without needing a network call on every search — faster, free, no
quota risk. An LLM-based extractor (extract_with_llm) is included as an
opt-in alternative for messier natural-language queries where regex
matching would miss a brand mentioned in an unusual way.

Blend score: `alpha * cosine(embedding) + (1 - alpha) * axis_similarity`,
where axis_similarity is 1 - normalized Euclidean distance between the
query's axis vector and each image's axis vector (so both terms are
similarities in [0, 1] and the blend is a straightforward weighted average).
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.txt"
AXIS_VECTORS_PATH = CACHE_DIR / "axis_vectors.npy"
PROBES_PATH = CACHE_DIR / "probes.joblib"
BRAND_VECTORS_PATH = CACHE_DIR / "brand_vectors.joblib"

# max possible Euclidean distance between two points in AXIS_SCALE^len(AXES)
_MAX_AXIS_DIST = np.sqrt(len(AXES) * (4**2))

# Words that qualify a brand ("early Stripe", "classic Nike") but carry no
# useful "vibe" meaning on their own — dropped from the leftover text so
# they don't get encoded as if they were a real style descriptor and dilute
# the brand-vector average with noise.
_FILLER_WORDS = {
    "early", "late", "classic", "old", "new", "modern", "current", "recent",
    "original", "vintage", "the", "a", "an", "of", "like", "style", "vibe",
    "vibes", "feel", "feels", "aesthetic", "era",
}


@dataclass
class SearchResult:
    image_id: str
    score: float
    cosine_sim: float
    axis_sim: float
    matched_brands: list[str]


def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class QueryEngine:
    def __init__(self):
        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.ids = IDS_PATH.read_text().splitlines()
        self.axis_vectors = np.load(AXIS_VECTORS_PATH)
        self.probes = joblib.load(PROBES_PATH)
        self.brand_vectors: dict = joblib.load(BRAND_VECTORS_PATH)
        self.brand_names = list(self.brand_vectors.keys())

        self._clip_model = None
        self._clip_tokenizer = None

    def _load_clip(self):
        if self._clip_model is None:
            model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
            self._clip_model = model.eval()
            self._clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        return self._clip_model, self._clip_tokenizer

    def parse_brands(self, query: str) -> tuple[list[str], str]:
        """Regex word-match known brand names out of the query, case-insensitive.
        Returns (matched brand names, remaining text with those names removed)."""
        matched = []
        remaining = query
        for brand in self.brand_names:
            pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
            if pattern.search(remaining):
                matched.append(brand)
                remaining = pattern.sub("", remaining)
        remaining = re.sub(r"\s+", " ", remaining).strip(" +,")

        # drop the remainder entirely if it's just filler words around the
        # brand mention(s) with no real descriptive content of its own
        leftover_words = {w.lower().strip(".,!?") for w in remaining.split()}
        if matched and leftover_words and leftover_words <= _FILLER_WORDS:
            remaining = ""

        return matched, remaining

    def extract_with_llm(self, query: str) -> tuple[list[str], str]:
        """Optional alternative to parse_brands for messier queries — asks
        Gemini to extract brand names + a cleaned adjective/vibe phrase.
        Not used by default; call explicitly when regex matching seems
        likely to miss something (e.g. a brand referred to indirectly)."""
        import json
        import os

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent / "data" / ".env")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set — can't use LLM extraction")

        from google import genai

        client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(timeout=30_000))
        prompt = f"""Known brand names: {", ".join(self.brand_names)}

Query: "{query}"

Which of the known brand names are mentioned in the query (match the brand
even if qualified, e.g. "early Stripe" still means "Stripe")? Also give the
remaining descriptive text with brand names removed (adjectives, vibe words).

Respond with ONLY JSON: {{"brands": [...], "remaining_text": "..."}}"""
        response = client.models.generate_content(model="gemini-3.5-flash-lite", contents=[prompt])
        text = response.text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        data = json.loads(text)
        brands = [b for b in data.get("brands", []) if b in self.brand_vectors]
        return brands, data.get("remaining_text", "").strip()

    def encode_text(self, text: str) -> np.ndarray:
        model, tokenizer = self._load_clip()
        with torch.no_grad():
            tokens = tokenizer([text])
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).numpy()

    def resolve_target(
        self, matched_brands: list[str], remaining_text: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine brand reference vectors and/or free-text CLIP encoding
        into one target embedding + one target axis vector."""
        embeddings, axis_vecs = [], []

        for brand in matched_brands:
            bv = self.brand_vectors[brand]
            embeddings.append(bv["embedding"])
            axis_vecs.append([bv["axis_scores"][a] for a in AXES])

        if remaining_text:
            text_embedding = self.encode_text(remaining_text)
            embeddings.append(text_embedding)
            text_axis = np.clip(
                [self.probes[a].predict(text_embedding.reshape(1, -1))[0] for a in AXES], 1, 5
            )
            axis_vecs.append(text_axis)

        if not embeddings:
            raise ValueError("query resolved to nothing — no brands matched and no free text left")

        target_embedding = np.mean(embeddings, axis=0)
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        target_axis = np.mean(axis_vecs, axis=0)
        return target_embedding, target_axis

    def search(self, query: str, alpha: float = 0.5, top_k: int = 20) -> list[SearchResult]:
        matched_brands, remaining_text = self.parse_brands(query)
        target_embedding, target_axis = self.resolve_target(matched_brands, remaining_text)

        cosine_sims = self.embeddings @ target_embedding
        axis_dists = np.linalg.norm(self.axis_vectors - target_axis, axis=1)
        axis_sims = 1 - (axis_dists / _MAX_AXIS_DIST)

        # CLIP's text->image cosine similarities sit on a much lower natural
        # scale than image->image ones (the "modality gap"), so a query with
        # any free text mixed in would otherwise make `alpha` nearly
        # meaningless — the axis term dominates regardless of its value.
        # Min-max normalizing both terms per-query, over this corpus, keeps
        # them comparable so alpha actually controls the blend.
        cosine_sims_norm = _normalize(cosine_sims)
        axis_sims_norm = _normalize(axis_sims)

        scores = alpha * cosine_sims_norm + (1 - alpha) * axis_sims_norm
        order = np.argsort(-scores)[:top_k]

        return [
            SearchResult(
                image_id=self.ids[i],
                score=float(scores[i]),
                cosine_sim=float(cosine_sims[i]),
                axis_sim=float(axis_sims[i]),
                matched_brands=matched_brands,
            )
            for i in order
        ]

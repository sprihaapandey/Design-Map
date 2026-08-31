"""Phase 6: FastAPI backend for the search UI.

Run with:
    uvicorn api.server:app --port 8000

Endpoints:
    GET  /api/axes            axis names + corpus min/max/mean (slider bounds)
    GET  /api/search           text query -> ranked results
    GET  /api/map              precomputed 2D layout of the whole corpus
    GET  /api/image/{id}       one image's metadata + axis scores
    POST /api/explore          nudge an image's vector along axis directions, re-query
    GET  /img/{id}.png         serves the actual screenshots (static)
"""

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.query import QueryEngine, _normalize
from config import AXES, CACHE_DIR, IMAGES_DIR
from labeling.db import get_labeling_conn

UI_DIR = Path(__file__).parent.parent / "ui"
UMAP_COORDS_PATH = CACHE_DIR / "umap_coords.npy"

app = FastAPI()
engine = QueryEngine()
umap_coords = np.load(UMAP_COORDS_PATH)

app.mount("/img", StaticFiles(directory=str(IMAGES_DIR)), name="img")


def _image_metadata() -> dict[str, dict]:
    conn = get_labeling_conn()
    rows = conn.execute("SELECT id, brand_name, source_url, style_tag, source_type FROM images").fetchall()
    conn.close()
    return {r[0]: {"brand_name": r[1], "source_url": r[2], "style_tag": r[3], "source_type": r[4]} for r in rows}


_METADATA = _image_metadata()


def _axis_scores_for(idx: int) -> dict[str, float]:
    return {axis: round(float(engine.axis_vectors[idx, i]), 2) for i, axis in enumerate(AXES)}


def _result_dict(image_id: str, score: float, cosine_sim: float, axis_sim: float) -> dict:
    idx = engine.ids.index(image_id) if image_id not in _ID_TO_IDX else _ID_TO_IDX[image_id]
    meta = _METADATA.get(image_id, {})
    return {
        "image_id": image_id,
        "score": round(score, 4),
        "cosine_sim": round(cosine_sim, 4),
        "axis_sim": round(axis_sim, 4),
        "axis_scores": _axis_scores_for(idx),
        **meta,
    }


_ID_TO_IDX = {image_id: i for i, image_id in enumerate(engine.ids)}


@app.get("/api/axes")
def get_axes():
    return {
        "axes": AXES,
        "ranges": {
            axis: {
                "min": round(float(engine.axis_vectors[:, i].min()), 2),
                "max": round(float(engine.axis_vectors[:, i].max()), 2),
                "mean": round(float(engine.axis_vectors[:, i].mean()), 2),
            }
            for i, axis in enumerate(AXES)
        },
        "brands": engine.brand_names,
    }


@app.get("/api/search")
def search(q: str = "", alpha: float = 0.5, top_k: int = 40, axis_overrides: str = ""):
    """axis_overrides: comma-separated "axis:value" pairs, e.g.
    "playfulness:4.5,density:1" — lets the axis sliders directly pin a
    target axis value on top of whatever the text query implies, so they
    can re-rank live without changing the query text. A slider left "off"
    doesn't appear here at all and the query-derived value is used as-is."""
    if not q.strip():
        raise HTTPException(400, "empty query")
    matched_brands, remaining_text = engine.parse_brands(q)
    try:
        target_embedding, target_axis = engine.resolve_target(matched_brands, remaining_text)
    except ValueError as e:
        raise HTTPException(400, str(e))

    target_axis = _apply_axis_overrides(target_axis, axis_overrides)

    results = _rank(target_embedding, target_axis, alpha, top_k)
    return {"matched_brands": matched_brands, "remaining_text": remaining_text, "results": results}


def _apply_axis_overrides(target_axis: np.ndarray, axis_overrides: str) -> np.ndarray:
    if not axis_overrides:
        return target_axis
    target_axis = target_axis.copy()
    for pair in axis_overrides.split(","):
        if ":" not in pair:
            continue
        axis, value = pair.split(":", 1)
        axis = axis.strip()
        if axis in AXES:
            target_axis[AXES.index(axis)] = float(value)
    return target_axis


def _rank(target_embedding: np.ndarray, target_axis: np.ndarray, alpha: float, top_k: int) -> list[dict]:
    cosine_sims = engine.embeddings @ target_embedding
    axis_dists = np.linalg.norm(engine.axis_vectors - target_axis, axis=1)
    axis_sims = 1 - (axis_dists / engine.axis_vectors.shape[1] ** 0.5 / 4)

    cosine_sims_norm = _normalize(cosine_sims)
    axis_sims_norm = _normalize(axis_sims)
    scores = alpha * cosine_sims_norm + (1 - alpha) * axis_sims_norm

    order = np.argsort(-scores)[:top_k]
    return [
        _result_dict(engine.ids[i], float(scores[i]), float(cosine_sims[i]), float(axis_sims[i]))
        for i in order
    ]


@app.get("/api/map")
def get_map():
    return {
        "points": [
            {
                "image_id": engine.ids[i],
                "umap_x": float(umap_coords[i, 0]),
                "umap_y": float(umap_coords[i, 1]),
                "axis_scores": _axis_scores_for(i),
                **_METADATA.get(engine.ids[i], {}),
            }
            for i in range(len(engine.ids))
        ]
    }


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    if image_id not in _ID_TO_IDX:
        raise HTTPException(404, "unknown image_id")
    idx = _ID_TO_IDX[image_id]
    return {"image_id": image_id, "axis_scores": _axis_scores_for(idx), **_METADATA.get(image_id, {})}


class ExploreRequest(BaseModel):
    image_id: str
    deltas: dict[str, float]  # axis -> nudge magnitude, e.g. {"playfulness": 1.5}
    alpha: float = 0.5
    top_k: int = 40


@app.post("/api/explore")
def explore(req: ExploreRequest):
    if req.image_id not in _ID_TO_IDX:
        raise HTTPException(404, "unknown image_id")
    base_idx = _ID_TO_IDX[req.image_id]
    base_embedding = engine.embeddings[base_idx].copy()

    nudged = base_embedding.copy()
    for axis, magnitude in req.deltas.items():
        if axis not in AXES or magnitude == 0:
            continue
        direction = engine.probes[axis].coef_
        direction = direction / np.linalg.norm(direction)
        nudged += magnitude * direction

    nudged = nudged / np.linalg.norm(nudged)
    nudged_axis = np.clip(
        [engine.probes[a].predict(nudged.reshape(1, -1))[0] for a in AXES], 1, 5
    )

    results = _rank(nudged, np.array(nudged_axis), req.alpha, req.top_k)
    return {
        "base_image_id": req.image_id,
        "nudged_axis_scores": {axis: round(float(v), 2) for axis, v in zip(AXES, nudged_axis)},
        "results": results,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (UI_DIR / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

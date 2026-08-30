"""Phase 3: minimal local web app for hand-labeling the seed set.

Run with:
    uvicorn labeling.label_app:app --port 8420

Then open http://localhost:8420 — it always jumps to the next seed image
that doesn't have a score for every axis yet, so labeling can be stopped
and resumed at any time.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AXES, AXIS_SCALE_MAX, AXIS_SCALE_MIN, IMAGES_DIR
from labeling.db import get_labeling_conn, seed_ids, upsert_label

app = FastAPI()
app.mount("/img", StaticFiles(directory=str(IMAGES_DIR)), name="img")


def next_unlabeled(conn) -> str | None:
    ids = seed_ids(conn)
    for image_id in ids:
        count = conn.execute("SELECT COUNT(*) FROM labels WHERE image_id = ?", (image_id,)).fetchone()[0]
        if count < len(AXES):
            return image_id
    return None


def progress(conn) -> tuple[int, int]:
    ids = seed_ids(conn)
    done = 0
    for image_id in ids:
        count = conn.execute("SELECT COUNT(*) FROM labels WHERE image_id = ?", (image_id,)).fetchone()[0]
        if count >= len(AXES):
            done += 1
    return done, len(ids)


def render_form(image_id: str, done: int, total: int) -> str:
    conn = get_labeling_conn()
    existing = dict(
        conn.execute("SELECT axis, score FROM labels WHERE image_id = ?", (image_id,)).fetchall()
    )
    conn.close()

    sliders = ""
    for axis in AXES:
        current = existing.get(axis, 3)
        sliders += f"""
        <div class="axis">
          <label for="{axis}">{axis} — <span id="{axis}_val">{current}</span></label>
          <input type="range" id="{axis}" name="{axis}" min="{AXIS_SCALE_MIN}" max="{AXIS_SCALE_MAX}"
                 value="{current}" oninput="document.getElementById('{axis}_val').innerText = this.value">
        </div>
        """

    return f"""
    <html>
    <head>
      <title>TasteMap labeling</title>
      <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; }}
        img {{ max-width: 100%; border: 1px solid #ddd; }}
        .axis {{ margin: 16px 0; }}
        label {{ display: block; margin-bottom: 4px; font-weight: 600; text-transform: capitalize; }}
        input[type=range] {{ width: 100%; }}
        button {{ margin-top: 24px; padding: 10px 24px; font-size: 16px; }}
        .progress {{ color: #666; margin-bottom: 16px; }}
      </style>
    </head>
    <body>
      <div class="progress">{done} / {total} labeled — {image_id}</div>
      <img src="/img/{image_id}.png">
      <form method="post" action="/label/{image_id}">
        {sliders}
        <button type="submit">Save &amp; Next</button>
      </form>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def index():
    conn = get_labeling_conn()
    image_id = next_unlabeled(conn)
    done, total = progress(conn)
    conn.close()
    if image_id is None:
        return f"<html><body><h2>All {total} seed images labeled.</h2></body></html>"
    return render_form(image_id, done, total)


@app.post("/label/{image_id}")
async def submit(image_id: str, request: Request):
    form = await request.form()
    conn = get_labeling_conn()
    for axis in AXES:
        if axis in form:
            upsert_label(conn, image_id, axis, int(form[axis]))
    conn.commit()
    conn.close()
    return RedirectResponse(url="/", status_code=303)

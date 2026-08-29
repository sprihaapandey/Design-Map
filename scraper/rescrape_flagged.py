"""Re-capture a specific set of already-registered image ids with the
cookie-banner-dismissing scraper, using each row's stored source_url. Used
after scan_overlays.py flags images where a consent dialog is covering the
page — re-scraping with the patch (see dismiss_cookie_banner in
scrape_sites.py) should recover a clean capture without one.
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGES_DIR
from scraper.db import get_conn, upsert_image
from scraper.scrape_sites import CONCURRENCY, NAV_TIMEOUT_MS, VIEWPORT, dismiss_cookie_banner

IDS_FILE = Path(__file__).parent.parent / "/tmp/cookie_flagged.txt"


async def recapture_one(browser, sem, row, results):
    async with sem:
        image_id, local_path, source_url, brand, style_tag, source_type = row
        dest = IMAGES_DIR.parent.parent / local_path
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
            await page.goto(source_url, wait_until="load")
            await page.wait_for_timeout(1000)
            await dismiss_cookie_banner(page)
            await page.screenshot(path=str(dest))
            await page.close()
            results.append((row, "ok"))
        except Exception as e:
            results.append((row, f"error: {e}"))


async def main():
    ids = [line.strip() for line in IDS_FILE.read_text().splitlines() if line.strip()]
    conn = get_conn()
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, local_path, source_url, brand_name, style_tag, source_type FROM images WHERE id IN ({placeholders})",
        ids,
    ).fetchall()

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        await asyncio.gather(*[recapture_one(browser, sem, row, results) for row in rows])
        await browser.close()

    from PIL import Image

    ok, failed = 0, 0
    for row, status in results:
        image_id, local_path, source_url, brand, style_tag, source_type = row
        if status == "ok":
            dest = IMAGES_DIR.parent.parent / local_path
            with Image.open(dest) as im:
                width, height = im.size
            upsert_image(
                conn,
                id=image_id,
                local_path=local_path,
                source_type=source_type,
                source_url=source_url,
                brand_name=brand,
                style_tag=style_tag,
                width=width,
                height=height,
            )
            ok += 1
        else:
            failed += 1
            print(f"  FAILED {image_id} ({source_url}): {status}")
    conn.commit()
    conn.close()
    print(f"done: {ok} recaptured, {failed} failed (of {len(rows)})")


if __name__ == "__main__":
    asyncio.run(main())

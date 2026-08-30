"""Phase 4: capture 1-2 extra pages per reference brand (beyond the homepage
already in the corpus) so the brand's reference vector is an average over
multiple pages rather than a single homepage screenshot — a homepage alone
can be unrepresentative (a big hero image, a promo banner) of a brand's
overall design language.

Tries a short list of common path candidates per brand and takes whichever
resolve; any that don't are just skipped (a brand with only 1 extra page,
or none, still works — the reference vector just averages fewer images).
"""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGES_DIR
from scraper.db import get_conn, upsert_image
from scraper.scrape_sites import CONCURRENCY, NAV_TIMEOUT_MS, VIEWPORT, dismiss_cookie_banner

BRANDS = [
    "Linear", "Arc", "Stripe", "Notion", "Vercel", "Figma", "Framer",
    "Superhuman", "Raycast", "Cron", "Airtable", "Webflow", "Airbnb",
    "Discord", "Spotify", "Slack", "Robinhood",
]

EXTRA_PATHS = ["/pricing", "/features"]


async def try_capture(browser, sem: asyncio.Semaphore, brand: str, homepage_url: str, path: str, results: list):
    url = homepage_url.rstrip("/") + path
    host = urlparse(homepage_url).netloc.replace("www.", "")
    image_id = "brandpage_" + host.replace(".", "-") + "_" + path.strip("/")
    dest = IMAGES_DIR / f"{image_id}.png"
    if dest.exists():
        results.append((image_id, dest, url, brand, "ok"))
        return
    async with sem:
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
            response = await page.goto(url, wait_until="load")
            if response is None or response.status >= 400:
                await page.close()
                results.append((image_id, None, url, brand, f"status {response.status if response else 'none'}"))
                return
            await page.wait_for_timeout(1000)
            await dismiss_cookie_banner(page)
            await page.screenshot(path=str(dest))
            await page.close()
            results.append((image_id, dest, url, brand, "ok"))
        except Exception as e:
            results.append((image_id, None, url, brand, f"error: {e}"))


async def main() -> None:
    conn = get_conn()
    rows = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT brand_name, source_url FROM images WHERE brand_name IN ({','.join('?' for _ in BRANDS)})",
            BRANDS,
        ).fetchall()
    }
    missing = [b for b in BRANDS if b not in rows]
    if missing:
        print(f"WARNING: no homepage found for {missing}")

    results: list = []
    sem = asyncio.Semaphore(CONCURRENCY)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        tasks = [
            try_capture(browser, sem, brand, rows[brand], path, results)
            for brand in BRANDS
            if brand in rows
            for path in EXTRA_PATHS
        ]
        await asyncio.gather(*tasks)
        await browser.close()

    from PIL import Image

    ok, failed = 0, 0
    for image_id, dest, url, brand, status in results:
        if status == "ok":
            with Image.open(dest) as im:
                width, height = im.size
            upsert_image(
                conn,
                id=image_id,
                local_path=str(dest.relative_to(IMAGES_DIR.parent.parent)),
                source_type="brand_page",
                source_url=url,
                brand_name=brand,
                style_tag=None,
                width=width,
                height=height,
            )
            ok += 1
        else:
            failed += 1
            print(f"  skip {url}: {status}")
    conn.commit()
    conn.close()
    print(f"done: {ok} captured, {failed} skipped (of {len(results)})")


if __name__ == "__main__":
    asyncio.run(main())

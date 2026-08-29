"""Phase 1: screenshot a curated list of real product/marketing sites with
Playwright, tagged by design style (see scraper/sites.csv), and register them
in the corpus DB.

Polite by default: bounded concurrency, per-page timeout, single viewport
screenshot (no deep crawling), skip-and-log on any failure rather than retry.
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGES_DIR
from scraper.db import get_conn, upsert_image

SITES_CSV = Path(__file__).parent / "sites.csv"
VIEWPORT = {"width": 1440, "height": 900}
CONCURRENCY = 6
NAV_TIMEOUT_MS = 20_000

# Common consent-banner accept buttons (OneTrust, Cookiebot, Osano, custom
# banners, ...) — best-effort click so the banner doesn't sit on top of the
# design in the screenshot. Playwright's :has-text() is a case-insensitive
# substring match, so a handful of keywords covers most phrasings (English
# "Accept"/"Accept All"/"Accept Optional Cookies"/"Accept & Close" etc. all
# match on "accept") plus a few major non-English markets seen in the corpus.
# Order matters: most specific/common first.
COOKIE_ACCEPT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "button#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "button[data-testid='uc-accept-all-button']",
    "button:has-text('accept')",
    "button:has-text('agree')",
    "button:has-text('allow all')",
    "button:has-text('got it')",
    "button:has-text('confirm my choices')",
    "button:has-text(\"i'm ok with that\")",
    "button:has-text('akzeptieren')",  # German
    "button:has-text('accepter')",  # French
    "button:has-text('aceptar')",  # Spanish
    "button:has-text('accetta')",  # Italian
    "button:has-text('akceptuj')",  # Polish
    "a:has-text('accept')",
]

# Fallback for promo/signup/region modals that aren't cookie banners at all
# (a "Get 20% off" popup, a country-redirect prompt, an onboarding tooltip) —
# these usually only offer a close icon, no "accept"-shaped button.
CLOSE_ICON_SELECTORS = [
    "[aria-label='Close' i]",
    "button.close",
    "[class*='close-button' i]",
    "[class*='modal-close' i]",
]

DISMISS_PASSES = 3


def slug_for(url: str) -> str:
    host = urlparse(url).netloc.replace("www.", "")
    return "scrape_" + host.replace(".", "-").replace("/", "-")


async def _try_click_one(page, selector: str) -> bool:
    try:
        locator = page.locator(selector).first
        if await locator.is_visible(timeout=600):
            await locator.click(timeout=600)
            await page.wait_for_timeout(900)
            return True
    except Exception:
        pass
    return False


async def dismiss_cookie_banner(page) -> None:
    # Multiple passes: sites often stack a survey/region modal *and* a
    # separate cookie bar, so one successful click doesn't mean we're done.
    for _ in range(DISMISS_PASSES):
        dismissed = False
        for selector in COOKIE_ACCEPT_SELECTORS:
            if await _try_click_one(page, selector):
                dismissed = True
                break
        if not dismissed:
            for selector in CLOSE_ICON_SELECTORS:
                if await _try_click_one(page, selector):
                    dismissed = True
                    break
        if not dismissed:
            break


async def screenshot_one(browser, sem: asyncio.Semaphore, row: dict, results: list):
    async with sem:
        url, brand, style_tag = row["url"], row["brand"], row["style_tag"]
        image_id = slug_for(url)
        dest = IMAGES_DIR / f"{image_id}.png"
        if dest.exists():
            results.append((image_id, dest, url, brand, style_tag, "cached"))
            return
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
            await page.goto(url, wait_until="load")
            await page.wait_for_timeout(1000)  # let hero animations/fonts settle
            await dismiss_cookie_banner(page)
            await page.screenshot(path=str(dest))
            await page.close()
            results.append((image_id, dest, url, brand, style_tag, "ok"))
        except Exception as e:
            results.append((image_id, None, url, brand, style_tag, f"error: {e}"))


async def main(limit: int | None) -> None:
    with open(SITES_CSV) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        await asyncio.gather(*[screenshot_one(browser, sem, row, results) for row in rows])
        await browser.close()

    conn = get_conn()
    ok, cached, failed = 0, 0, 0
    from PIL import Image

    for image_id, dest, url, brand, style_tag, status in results:
        if status == "ok" or status == "cached":
            with Image.open(dest) as im:
                width, height = im.size
            upsert_image(
                conn,
                id=image_id,
                local_path=str(dest.relative_to(IMAGES_DIR.parent.parent)),
                source_type="curated_scrape",
                source_url=url,
                brand_name=brand,
                style_tag=style_tag,
                width=width,
                height=height,
            )
            ok += status == "ok"
            cached += status == "cached"
        else:
            failed += 1
            print(f"  FAILED {url}: {status}")
    conn.commit()
    conn.close()
    print(f"done: {ok} new, {cached} cached, {failed} failed (of {len(rows)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.limit))

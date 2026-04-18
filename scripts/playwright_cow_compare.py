#!/usr/bin/env python3
"""
playwright_cow_compare.py — W1z4rD V1510n visual feedback loop.

Takes screenshots of the 3D world and the video sensor feed,
compares entity counts and visual similarity, and reports pass/fail.

Requirements:
  pip install playwright Pillow
  playwright install chromium

Usage:
  python scripts/playwright_cow_compare.py [--iterations 5] [--url http://localhost:8093]
  python scripts/playwright_cow_compare.py --once      # single capture + report
"""
import argparse, base64, io, json, math, os, sys, time
from pathlib import Path

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Deps ─────────────────────────────────────────────────────────────────────

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright not installed.  Run:  pip install playwright && playwright install chromium")
    raise SystemExit(1)

try:
    from PIL import Image, ImageChops, ImageFilter
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] Pillow not installed — visual diff disabled.  pip install Pillow")

OUT_DIR = Path(__file__).resolve().parent.parent / "playwright_captures"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pil_from_b64(b64_str: str) -> "Image.Image":
    data = base64.b64decode(b64_str.split(",", 1)[-1])
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pixel_similarity(img_a: "Image.Image", img_b: "Image.Image") -> float:
    """Mean-squared pixel difference normalised to [0,1] similarity."""
    a = img_a.resize((160, 90), Image.BILINEAR)
    b = img_b.resize((160, 90), Image.BILINEAR)
    diff = ImageChops.difference(a, b)
    pixels = list(diff.getdata())
    mse = sum(((r + g + b) / 3) ** 2 for r, g, b in pixels) / (len(pixels) * 255 ** 2)
    return max(0.0, 1.0 - math.sqrt(mse) * 4)


def _count_non_green_blobs(img: "Image.Image") -> int:
    """Rough entity count via column projection (same algorithm as sensor stream)."""
    W, H = 96, 54
    small = img.resize((W, H), Image.BILINEAR)
    col_counts = []
    for x in range(W):
        count = 0
        for y in range(H // 4, H):
            r, g, b = small.getpixel((x, y))
            if not (g > r + 14 and g > b + 8 and g > 50):
                count += 1
        col_counts.append(count)
    THRESH = (H * 3 // 4) // 8
    blobs, in_run = 0, False
    for cnt in col_counts:
        if cnt >= THRESH:
            if not in_run:
                blobs += 1; in_run = True
        else:
            in_run = False
    return blobs


# ─── Main comparison run ──────────────────────────────────────────────────────

def run_comparison(url: str, iteration: int, page) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wait for the JS helpers to be defined (page rendered)
    page.wait_for_function("typeof window.__captureFrame === 'function'", timeout=15_000)
    # Wait for at least one entity (old sensor stream fallback spawns after first WS msg)
    try:
        page.wait_for_function("window.__getCowCount && window.__getCowCount() > 0", timeout=25_000)
    except Exception:
        pass  # continue anyway — will report 0 cows

    # Settle for texture / position update
    page.wait_for_timeout(1500)

    # Capture 3D render
    render_b64 = page.evaluate("window.__captureFrame()")
    render_path = OUT_DIR / f"iter{iteration:03d}_render.png"
    render_img  = _pil_from_b64(render_b64) if PIL_OK else None
    if render_img:
        render_img.save(render_path)

    # Capture video feed thumbnail
    feed_src = page.evaluate("window.__getFrameData()")
    feed_img = None
    feed_path = OUT_DIR / f"iter{iteration:03d}_feed.png"
    if feed_src and PIL_OK:
        feed_img = _pil_from_b64(feed_src)
        feed_img.save(feed_path)

    # Entity counts
    cow_count_3d  = page.evaluate("window.__getCowCount()")
    cow_count_vid = _count_non_green_blobs(feed_img) if (PIL_OK and feed_img) else None

    # Visual similarity between render and video frame
    sim_score = None
    if PIL_OK and render_img and feed_img:
        # Crop the render to remove the overlay area (right 220px contains the HUD)
        w3d, h3d = render_img.size
        render_crop = render_img.crop((0, 0, w3d - 240, h3d))
        feed_resized = feed_img.resize(render_crop.size, Image.BILINEAR)
        sim_score = _pixel_similarity(render_crop, feed_resized)

    result = {
        "iteration":     iteration,
        "cow_count_3d":  cow_count_3d,
        "cow_count_vid": cow_count_vid,
        "sim_score":     round(sim_score, 3) if sim_score is not None else None,
        "render_path":   str(render_path),
        "feed_path":     str(feed_path) if feed_img else None,
        "pass":          (
            cow_count_3d > 0
            and (cow_count_vid is None or abs(cow_count_3d - cow_count_vid) <= 1)
            and (sim_score is None or sim_score > 0.30)
        ),
    }

    print(
        f"  iter {iteration:3d}  |"
        f"  3D cows={cow_count_3d}"
        f"  vid cows={cow_count_vid if cow_count_vid is not None else '?'}"
        f"  sim={result['sim_score'] if result['sim_score'] is not None else '?'}"
        f"  → {'PASS ✓' if result['pass'] else 'FAIL ✗'}"
    )
    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="W1z4rD cow world visual feedback loop")
    ap.add_argument("--url",        default="http://localhost:8093/?screenshot=1")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--once",       action="store_true", help="Run a single iteration")
    ap.add_argument("--interval",   type=float, default=4.0, help="Seconds between iterations")
    args = ap.parse_args()

    iters = 1 if args.once else args.iterations

    print("=" * 60)
    print("  W1z4rD Playwright Feedback Loop")
    print(f"  URL: {args.url}")
    print(f"  Iterations: {iters}")
    print("=" * 60)

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx     = browser.new_context(viewport={"width": 1280, "height": 720})
        page    = ctx.new_page()
        page.goto(args.url)

        # Wait for Three.js to initialise (canvas in DOM + expose functions)
        page.wait_for_function("typeof window.__captureFrame === 'function'", timeout=15_000)
        print("  [OK] Page loaded, waiting for first WS frame...")

        for i in range(1, iters + 1):
            try:
                r = run_comparison(args.url, i, page)
                results.append(r)
                if r["pass"] and i > 1:
                    # Two consecutive passes = done
                    if results[-2]["pass"]:
                        print("\n  ✓ Two consecutive passes — scene looks correct.")
                        break
            except Exception as ex:
                print(f"  iter {i}: ERROR — {ex}")
            if i < iters:
                time.sleep(args.interval)

        browser.close()

    # Summary report
    print("\n-- Summary ------------------------------------------------------")
    passes = sum(1 for r in results if r["pass"])
    print(f"  {passes}/{len(results)} iterations passed")
    if results:
        avg_sim = [r["sim_score"] for r in results if r["sim_score"] is not None]
        if avg_sim:
            print(f"  avg similarity: {sum(avg_sim)/len(avg_sim):.3f}")
    print(f"  screenshots saved to: {OUT_DIR}")
    print("-----------------------------------------------------------------")

    # Write JSON report
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()

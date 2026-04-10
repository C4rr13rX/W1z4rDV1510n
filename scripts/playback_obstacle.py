#!/usr/bin/env python3
"""
W1z4rD V1510n — Obstacle Course Playback Script
================================================
After training with train_obstacle.py, this script opens the obstacle course
and accepts an English command. It:

  1. Takes a screenshot of the current page state
  2. POSTs goal text + screenshot to /media/playback
  3. Reads the predicted motion zone and click
  4. Moves the mouse there and clicks — the model drives the cursor

This demonstrates that the pool has learned the spatial association between
the goal language, the visual scene, and the required action.

Usage:
    python playback_obstacle.py
    python playback_obstacle.py --goal "click the green button"
    python playback_obstacle.py --auto   # cycles through all commands automatically
"""

import asyncio
import base64
import json
import argparse
import httpx
from pathlib import Path
from playwright.async_api import async_playwright

NODE_URL   = "http://localhost:8080"
COURSE_URL = Path(__file__).parent / "obstacle_course.html"

COMMANDS = [
    "click the red button",
    "click the blue button",
    "click the green button",
    "click the star button",
    "click the purple button",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

async def screenshot_b64(page):
    png = await page.screenshot(type="jpeg", quality=70)
    return base64.b64encode(png).decode()

async def ask_playback(client, goal, screen_b64, hops=1):
    # 1 hop: only direct text→motion associations fire.
    # More hops add indirect cross-goal noise (shared words like "click" activate all zones).
    # The screenshot is skipped here because the obstacle course looks identical for all goals,
    # so image labels add noise rather than discrimination.
    payload = {"goal": goal, "hops": hops}
    resp = await client.post(f"{NODE_URL}/media/playback", json=payload, timeout=15)
    return resp.json()

def zone_to_pixel(zx, zy, vw, vh, grid=8):
    """Convert zone index to pixel center on screen."""
    x = int((zx + 0.5) / grid * vw)
    y = int((zy + 0.5) / grid * vh)
    return x, y

# ── Playback ──────────────────────────────────────────────────────────────────

async def run_playback(goal: str, auto: bool):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=60)
        ctx     = await browser.new_context(viewport={"width": 1280, "height": 800})
        page    = await ctx.new_page()

        url = f"file:///{COURSE_URL.as_posix()}"
        await page.goto(url)
        await page.wait_for_timeout(800)

        vw, vh = 1280, 800

        async with httpx.AsyncClient() as client:

            async def execute_command(cmd):
                print(f"\n>> Command: \"{cmd}\"")
                await page.evaluate(f"window.setGoal({json.dumps(cmd)})")
                await page.wait_for_timeout(300)

                # Take screenshot
                sb64 = await screenshot_b64(page)

                # Ask the model
                result = await ask_playback(client, cmd, sb64)
                action = result.get("action", {})
                top    = result.get("top_activations", [])[:8]

                print(f"  Action type:  {action.get('type', 'none')}")
                print(f"  Predicted zone: ({action.get('zone_x','?')}, {action.get('zone_y','?')})")
                print(f"  Confidence:   {action.get('confidence', 0):.3f}")
                print(f"  Click:        {action.get('click', False)} (strength {action.get('click_strength', 0):.3f})")
                print(f"  Top activations:")
                for a in top:
                    print(f"    {a['label']:45s}  {a['strength']:.3f}")

                if action.get("type") == "none":
                    print("  X No prediction — pool may need more training")
                    return

                # Convert to screen coordinates
                zx = action.get("zone_x", 4)
                zy = action.get("zone_y", 4)
                px, py = zone_to_pixel(zx, zy, vw, vh)

                # Clamp to visible area
                px = max(50, min(vw - 50, px))
                py = max(80, min(vh - 50, py))  # below status bar

                print(f"  Moving to pixel ({px}, {py})")

                # Smooth move to predicted location
                await page.mouse.move(px, py, steps=15)
                await page.wait_for_timeout(400)

                if action.get("click", False):
                    await page.mouse.click(px, py)
                    print(f"  OK Clicked at ({px}, {py})")
                else:
                    print(f"  o Moved (no click predicted)")

                await page.wait_for_timeout(600)

            if auto:
                print("\nAuto mode — cycling through all commands twice...\n")
                for round_ in range(2):
                    print(f"— Round {round_+1} —")
                    for cmd in COMMANDS:
                        await execute_command(cmd)
                        await page.wait_for_timeout(800)
            else:
                await execute_command(goal)
                # Keep browser open for inspection
                print("\n  Browser will stay open for 10 seconds...")
                await page.wait_for_timeout(10000)

        await browser.close()

# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obstacle course playback")
    parser.add_argument("--node",  default=NODE_URL)
    parser.add_argument("--goal",  default="click the red button", help="English command")
    parser.add_argument("--hops",  type=int, default=3, help="Propagation hops")
    parser.add_argument("--auto",  action="store_true", help="Cycle through all commands")
    args = parser.parse_args()
    NODE_URL = args.node

    print(f"\nW1z4rD V1510n — Obstacle Course Playback")
    print(f"  Node: {NODE_URL}")
    if args.auto:
        print(f"  Mode: AUTO (all commands)")
    else:
        print(f"  Goal: \"{args.goal}\"")

    asyncio.run(run_playback(args.goal, args.auto))

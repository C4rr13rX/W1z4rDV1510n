#!/usr/bin/env python3
"""
W1z4rD V1510n -- Obstacle Course Training Script
================================================
Uses Playwright to open the obstacle course, record training interactions,
and POST them to the running node via /media/train.

Each training example is:
  - A screenshot of the page (image labels)
  - The English goal text (text labels)
  - The mouse trajectory from start -> target (motion labels)
  - A click event at the target (action label)

All four modalities are co-activated in one training call so the pool
learns that "click the red button" + red-button-zone-image + trajectory
+ click all belong together.

Usage:
    pip install playwright
    playwright install chromium
    python train_obstacle.py [--node http://localhost:8090] [--reps 8]
"""

import asyncio
import base64
import json
import math
import random
import sys
import time
import argparse
import httpx
from pathlib import Path
from playwright.async_api import async_playwright

NODE_URL   = "http://localhost:8090"
COURSE_URL = Path(__file__).parent / "obstacle_course.html"

TARGETS = {
    # button_id -> (full goal phrase, unique discriminative word)
    "btn-red":    ("click the red button",    "red"),
    "btn-blue":   ("click the blue button",   "blue"),
    "btn-green":  ("click the green button",  "green"),
    "btn-star":   ("click the star button",   "star"),
    "btn-purple": ("click the purple button", "purple"),
}

# -- Helpers -------------------------------------------------------------------

def screen_frac(px, py, vw, vh):
    """Convert pixel coords to [0,1] fractions."""
    return max(0.0, min(1.0, px / vw)), max(0.0, min(1.0, py / vh))

def build_trajectory(sx, sy, tx, ty, vw, vh, steps=12, duration_secs=0.8):
    """Interpolate a slightly curved path from (sx,sy) -> (tx,ty)."""
    pts = []
    # Add a small midpoint offset for natural-looking movement
    mx = (sx + tx) / 2 + random.uniform(-30, 30)
    my = (sy + ty) / 2 + random.uniform(-20, 20)
    for i in range(steps + 1):
        t = i / steps
        # Quadratic bezier: start -> mid -> end
        x = (1-t)**2 * sx + 2*(1-t)*t * mx + t**2 * tx
        y = (1-t)**2 * sy + 2*(1-t)*t * my + t**2 * ty
        fx, fy = screen_frac(x, y, vw, vh)
        pts.append({"x": fx, "y": fy, "t_secs": t * duration_secs, "click": False})
    # Final point = click
    fx, fy = screen_frac(tx, ty, vw, vh)
    pts.append({"x": fx, "y": fy, "t_secs": duration_secs + 0.05, "click": True})
    return pts

async def take_screenshot_b64(page):
    png = await page.screenshot(type="jpeg", quality=70)
    return base64.b64encode(png).decode()

async def post_training(client, goal, screen_b64, motion):
    payload = {
        "modality": "full",
        "data_b64": screen_b64,
        "text": goal,
        "motion": motion,
        "lr_scale": 1.0,
    }
    resp = await client.post(f"{NODE_URL}/media/train", json=payload, timeout=15)
    return resp.json()

async def post_anchor(client, unique_word, endpoint_motion, lr_scale=5.0):
    """Focused training: unique word + endpoint only -- no shared words, no trajectory.

    This creates a HIGH-WEIGHT direct link between the discriminative word
    and the target zone, counteracting the noise from shared words in the full
    trajectory training.  lr_scale=5.0 makes this 5x stronger than the full
    trajectory examples.
    """
    payload = {
        "modality": "full",
        "text": unique_word,       # just the color/object word, NOT "click the X button"
        "motion": endpoint_motion, # just the single endpoint click point
        "lr_scale": lr_scale,
    }
    resp = await client.post(f"{NODE_URL}/media/train", json=payload, timeout=15)
    return resp.json()

# -- Main training loop --------------------------------------------------------

async def train(reps: int):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx     = await browser.new_context(viewport={"width": 1280, "height": 800})
        page    = await ctx.new_page()

        url = f"file:///{COURSE_URL.as_posix()}"
        await page.goto(url)
        await page.wait_for_timeout(800)

        vw = 1280
        vh = 800

        async with httpx.AsyncClient() as client:
            total = 0
            for rep in range(reps):
                # Shuffle target order each rep for variety
                targets = list(TARGETS.items())
                random.shuffle(targets)

                for target_id, (goal, unique_word) in targets:
                    # Set goal display
                    await page.evaluate(f"window.setGoal({json.dumps(goal)})")
                    await page.wait_for_timeout(200)

                    # Get target center
                    center = await page.evaluate(f"window.getTargetCenter({json.dumps(target_id)})")
                    if not center:
                        print(f"  SKIP: {target_id} not found")
                        continue

                    tx, ty = center["x"], center["y"]

                    # Random start position (not on the target)
                    sx = random.uniform(100, vw - 100)
                    sy = random.uniform(100 + 48, vh - 100)  # below status bar
                    # Make sure start isn't too close to target
                    while math.hypot(sx - tx, sy - ty) < 200:
                        sx = random.uniform(100, vw - 100)
                        sy = random.uniform(100 + 48, vh - 100)

                    # Move mouse to start
                    await page.mouse.move(sx, sy)
                    await page.wait_for_timeout(100)

                    # Take pre-click screenshot
                    screen_b64 = await take_screenshot_b64(page)

                    # Build trajectory
                    motion = build_trajectory(sx, sy, tx, ty, vw, vh)

                    # Actually move the mouse (for realism)
                    for pt in motion[:-1]:
                        px = pt["x"] * vw
                        py = pt["y"] * vh
                        await page.mouse.move(px, py)
                        await asyncio.sleep(0.03)

                    # Click
                    await page.mouse.click(tx, ty)
                    await page.wait_for_timeout(150)

                    # Build the endpoint-only motion (just the final click, no path)
                    fx, fy = screen_frac(tx, ty, vw, vh)
                    endpoint_motion = [{"x": fx, "y": fy, "t_secs": 0.0, "click": True}]

                    # POST full trajectory (rich multi-modal context)
                    try:
                        result = await post_training(client, goal, screen_b64, motion)
                        lc = result.get("label_count", "?")
                        print(f"  Rep {rep+1:2d} | {goal:35s} | {lc} labels (full)")
                    except Exception as e:
                        print(f"  ERROR posting full training: {e}")

                    # POST anchor: unique word + endpoint only (5x weight for discrimination)
                    try:
                        anchor = await post_anchor(client, unique_word, endpoint_motion)
                        alc = anchor.get("label_count", "?")
                        total += 1
                        print(f"         | anchor: '{unique_word}' to zone   | {alc} labels (anchor)")
                    except Exception as e:
                        print(f"  ERROR posting anchor: {e}")

                    await page.wait_for_timeout(300)

            print(f"\nOK Training complete -- {total} anchor pairs across {reps} reps")
            print(f"  Each rep: {len(TARGETS)} full trajectories + {len(TARGETS)} anchor pairs")
            print(f"  Run playback_obstacle.py to test recall.\n")

        await browser.close()

# -- Entry ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train obstacle course")
    parser.add_argument("--node", default=NODE_URL, help="Node base URL")
    parser.add_argument("--reps", type=int, default=8, help="Training repetitions")
    args = parser.parse_args()
    NODE_URL = args.node

    print(f"\nW1z4rD V1510n -- Obstacle Course Trainer")
    print(f"  Node:  {NODE_URL}")
    print(f"  Reps:  {args.reps}")
    print(f"  Targets: {len(TARGETS)} buttons x {args.reps} reps")
    print(f"  Per rep: 1 full trajectory + 1 anchor pair = {len(TARGETS)*2*args.reps} total training calls\n")

    asyncio.run(train(args.reps))

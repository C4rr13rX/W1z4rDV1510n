#!/usr/bin/env python3
"""Playwright screenshot test for the photorealistic world viewer."""
import sys, os, time
from playwright.sync_api import sync_playwright

URL   = 'http://localhost:9000/world_viewer.html?vq=2'  # MEDIUM quality (stride=3)
OUT   = 'D:/w1z4rdv1510n-data/screenshots'
os.makedirs(OUT, exist_ok=True)

def shoot(page, name, delay=0):
    if delay: time.sleep(delay)
    path = f'{OUT}/{name}.png'
    page.screenshot(path=path, full_page=False)
    print(f'  saved {path}')
    return path

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-web-security',       # allow cross-origin fetch to :9001
            '--disable-features=VizDisplayCompositor',
            '--use-gl=swiftshader',         # software WebGL
        ]
    )
    ctx  = browser.new_context(viewport={'width': 1280, 'height': 720})
    page = ctx.new_page()

    # Capture console errors
    errors = []
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: errors.append(str(e)))

    print(f'Loading {URL} ...')
    page.goto(URL, wait_until='domcontentloaded', timeout=20000)

    # Screenshot 1: just after load (loading screen)
    shoot(page, '01_initial')

    # Wait for Three.js init + auto-mode detection
    print('Waiting for init...')
    time.sleep(8)
    shoot(page, '02_after_init')

    # Check if VIDEO button is active
    vbtn = page.query_selector('#video-btn')
    if vbtn:
        cls = vbtn.get_attribute('class') or ''
        print(f'VIDEO btn class: "{cls}"')
        if 'active' not in cls:
            print('  -> Clicking VIDEO button manually')
            vbtn.click()
            time.sleep(4)
            shoot(page, '03_video_clicked')

    # Wait for point cloud to build
    time.sleep(5)
    shoot(page, '04_point_cloud')

    # Check quality HUD
    fps_el = page.query_selector('#vq-fps')
    if fps_el:
        print(f'FPS display: {fps_el.inner_text()}')
    qual_el = page.query_selector('#vq-qual')
    if qual_el:
        print(f'Quality: {qual_el.inner_text()}')
    pts_el = page.query_selector('#vq-pts')
    if pts_el:
        print(f'Points: {pts_el.inner_text()}')

    # Wait more and take final
    time.sleep(5)
    shoot(page, '05_final')

    # ── NEURAL mode: click ◈ NEURAL button and capture the 3D cow ─────────
    print('\nSwitching to NEURAL mode...')
    neural_btn = page.query_selector('#neural-btn')
    if neural_btn:
        neural_btn.click()
        time.sleep(2)
        shoot(page, '06_neural_init')

        # Wait for cow geometry to fetch and render
        time.sleep(6)
        shoot(page, '07_neural_cow')

        # Check HUD info
        for hud_id in ('neural-status', 'neural-pose'):
            el = page.query_selector(f'#{hud_id}')
            if el: print(f'  {hud_id}: {el.inner_text()}')

        # Orbit slightly: drag horizontally to see cow from a 3/4 view
        canvas = page.query_selector('canvas')
        if canvas:
            bb = canvas.bounding_box()
            cx = bb['x'] + bb['width'] / 2
            cy = bb['y'] + bb['height'] / 2
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx - 120, cy - 30)
            page.mouse.up()
            time.sleep(1)
            shoot(page, '08_neural_orbit')
    else:
        print('  neural-btn not found')

    if errors:
        print(f'\nConsole errors ({len(errors)}):')
        for e in errors[:10]: print(f'  {e}')

    browser.close()
    print('\nDone.')

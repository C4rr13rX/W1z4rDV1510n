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

    if errors:
        print(f'\nConsole errors ({len(errors)}):')
        for e in errors[:10]: print(f'  {e}')

    browser.close()
    print('\nDone.')

"""Headless screenshot of the local site, never the owner's own browser.
usage: python3 scripts/shot.py URL OUT.png [--scroll-to SELECTOR | --y PIXELS] [--progress SELECTOR:FRACTION] [--wait MS] [--width 1440 --height 900]
--progress scrolls so that a tall section is FRACTION of the way through its own scroll range."""
import sys, argparse
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument('url'); ap.add_argument('out')
ap.add_argument('--scroll-to'); ap.add_argument('--y', type=int); ap.add_argument('--progress'); ap.add_argument('--wait', type=int, default=2500)
ap.add_argument('--width', type=int, default=1440); ap.add_argument('--height', type=int, default=900)
a = ap.parse_args()
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=True, args=['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader'])
    pg = b.new_page(viewport={'width': a.width, 'height': a.height}, device_scale_factor=1)
    logs = []; pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}') if m.type in ('error', 'warning') else None); pg.on('pageerror', lambda e: logs.append(f'pageerror: {e}'))
    pg.goto(a.url, wait_until='networkidle'); pg.wait_for_timeout(1500)
    if a.scroll_to: pg.evaluate("s => document.querySelector(s).scrollIntoView({block:'start'})", a.scroll_to)
    if a.y is not None: pg.evaluate("y => window.scrollTo(0, y)", a.y)
    if a.progress:
        sel, f = a.progress.rsplit(':', 1)
        pg.evaluate("([s, f]) => { const el = document.querySelector(s); const top = el.getBoundingClientRect().top + scrollY; window.scrollTo(0, top + (el.offsetHeight - innerHeight) * f); }", [sel, float(f)])
    pg.wait_for_timeout(a.wait); pg.screenshot(path=a.out)
    print('scrollWidth', pg.evaluate('document.documentElement.scrollWidth'), 'innerWidth', pg.evaluate('innerWidth'))
    for l in logs[:20]: print(l)
    b.close()

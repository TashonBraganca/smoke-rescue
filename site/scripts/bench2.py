"""Steady-state frame cost per place on the page. Parks at each spot, waits for loading, times frames for 3 s.
usage: SOFTGL=1 python3 scripts/bench2.py URL [URL2 ...] [--phone]"""
import argparse, os
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument('urls', nargs='+'); ap.add_argument('--phone', action='store_true'); a = ap.parse_args()
SPOTS = [('hero', 'top', 0.0), ('why', 'top', 0.06), ('lidar', 'top', 0.22), ('face', 'top', 0.43), ('eye-col', 'top', 0.52), ('eye-hot', 'top', 0.67), ('eye-per', 'top', 0.77),
         ('fade-out', 'top', 0.805), ('two dogs', 'top', 0.9), ('crew', 'top', 0.985),
         ('mission', 'mission', None), ('smoke', 'smoke', None), ('team', 'team', None), ('house', 'house', None), ('built', 'built', None)]
GL = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] if os.environ.get('SOFTGL') else ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist']
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=True, args=GL)
    rows = {}
    for url in a.urls:
        ctx = b.new_context(viewport={'width': 390, 'height': 844} if a.phone else {'width': 1440, 'height': 900}, device_scale_factor=3 if a.phone else 2, is_mobile=a.phone, has_touch=a.phone)
        pg = ctx.new_page(); pg.goto(url, wait_until='networkidle'); pg.wait_for_timeout(3000)
        coarse = pg.evaluate("matchMedia('(pointer: coarse)').matches")
        for name, sec, f in SPOTS:
            pg.evaluate("""([s, f]) => { const el = document.getElementById(s); const top = el.getBoundingClientRect().top + scrollY;
              window.scrollTo(0, f === null ? top : top + (el.offsetHeight - innerHeight) * f); }""", [sec, f])
            pg.wait_for_timeout(2500)
            ms = pg.evaluate("""() => new Promise(done => { const out = []; let last = performance.now(); const t0 = last;
              const step = now => { out.push(now - last); last = now; if (now - t0 < 3000) requestAnimationFrame(step); else done(out); }; requestAnimationFrame(step); })""")
            ms = sorted(ms[1:]); n = len(ms)
            rows.setdefault(name, []).append(f"{1000 * n / sum(ms):5.1f} fps  med {ms[n // 2]:6.1f}  p95 {ms[int(n * .95)]:6.1f}")
        print(url, 'coarse pointer:', coarse)
        ctx.close()
    for name, v in rows.items(): print(name.ljust(9), ' | '.join(v))
    b.close()

"""Scroll the whole page at a steady pace and record frame times.
usage: python3 scripts/bench.py URL [--phone] [--cpu 4] [--seconds 24]
Prints per-section median / p95 frame time and the share of frames over 33 ms (visible stutter) and over 50 ms."""
import argparse, json
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument('url'); ap.add_argument('--phone', action='store_true'); ap.add_argument('--cpu', type=float, default=1); ap.add_argument('--seconds', type=float, default=24)
a = ap.parse_args()
with sync_playwright() as p:
    import os
    GL = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] if os.environ.get('SOFTGL') else ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader']
    b = p.chromium.launch(channel='chrome', headless=True, args=GL)
    ctx = b.new_context(viewport={'width': 390, 'height': 844} if a.phone else {'width': 1440, 'height': 900}, device_scale_factor=3 if a.phone else 2, is_mobile=a.phone, has_touch=a.phone)
    pg = ctx.new_page(); cdp = ctx.new_cdp_session(pg)
    if a.cpu > 1: cdp.send('Emulation.setCPUThrottlingRate', {'rate': a.cpu})
    pg.goto(a.url, wait_until='networkidle'); pg.wait_for_timeout(3000)
    res = pg.evaluate("""async (secs) => {
      const ids = ['top','mission','smoke','team','house','built'];
      const secAt = () => { let s = 'top'; for (const id of ids) { const el = document.getElementById(id); if (el && el.getBoundingClientRect().top < innerHeight / 2) s = id; } return s; };
      const H = document.documentElement.scrollHeight - innerHeight, T = secs * 1000, out = [];
      let last = performance.now(); const t0 = last;
      await new Promise(done => { const step = (now) => { out.push([now - last, secAt()]); last = now; const f = Math.min(1, (now - t0) / T); window.scrollTo(0, H * f); if (f < 1) requestAnimationFrame(step); else done(); }; requestAnimationFrame(step); });
      return out;
    }""", a.seconds)
    by = {}
    for dt, s in res[2:]: by.setdefault(s, []).append(dt)
    allf = [d for d, _ in res[2:]]
    def stats(v):
        v = sorted(v); n = len(v); return dict(n=n, med=round(v[n // 2], 1), p95=round(v[int(n * .95)], 1), over33=round(100 * sum(x > 33.4 for x in v) / n, 1), over50=round(100 * sum(x > 50 for x in v) / n, 1))
    print('ALL', stats(allf), 'fps', round(1000 * len(allf) / sum(allf), 1))
    for s in ['top', 'mission', 'smoke', 'team', 'house', 'built']:
        if s in by: print(s.ljust(8), stats(by[s]))
    b.close()

// Test the unchanged production page with real replay files and a controlled live refusal.
const fs = require('fs'), path = require('path'), assert = require('assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/private/tmp/ryor-browser/node_modules/playwright');
const ROOT = path.resolve(__dirname, '../..'), OUT = path.join(ROOT, 'handoff/backend_evidence');
const URL = process.env.RYOR_URL || 'http://host.docker.internal:8204';
(async () => {
  const browser = await chromium.connect(process.env.BROWSER_WS);
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const errors = [], checks = [], refusals = [];
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  // Do not launch paid AWS tasks to test a static replay. Only this transport is simulated.
  await context.route('**/api/mission', async route => {
    const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'POST, OPTIONS' };
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
    const setup = route.request().postDataJSON();
    refusals.push(setup);
    await route.fulfill({ headers, json: { source: 'recorded', reason: 'disabled', setup: `${setup.house}__r${setup.robots}__easy` } });
  });
  try {
    const response = await context.request.get(URL + '/rescue-data/gallery.json');
    assert(response.ok());
    const gallery = await response.json();
    assert.equal(gallery.houses.length, 5);
    for (const house of gallery.houses) {
      for (const n of [1, 2, 3]) {
        const key = `${house.id}__r${n}__easy`, available = house.available_replays.includes(n);
        const file = await context.request.get(`${URL}/rescue-data/${key}.json`);
        assert.equal(file.status(), available ? 200 : 404, key);
        if (available) {
          const run = await file.json();
          assert(run.validated && run.gate.all_rooms_entered && run.gate.all_out && run.gate.false_confirmations === 0);
          assert.equal(run.frames.length, run.duration_s * 2 + 1);
        }
        // The three project houses use their films. Test their replay files above without changing page flow.
        if (house.source === 'ours') { checks.push({ key, available, check: 'static HTTP and payload' }); continue; }
        await page.goto(`${URL}/rescue/?house=${house.id}&robots=${n}&start=1`);
        await page.getByRole('button', { name: 'Skip cinematic intro', exact: true }).click();
        if (!available) {
          await page.getByRole('region', { name: 'Live rescue', exact: true }).getByRole('alert').waitFor();
          assert.equal(await page.locator('[data-setup]').count(), 0);
          checks.push({ key, available, check: 'failed replay withheld; unavailable message displayed' });
          continue;
        }
        await page.waitForSelector(`[data-setup="${key}"]`);
        await page.waitForFunction(() => Number(document.querySelector('[data-first-frame]')?.getAttribute('data-first-frame')) > 0);
        assert((await page.locator('[data-source]').innerText()).includes('Recorded replay'));
        const slider = page.getByRole('slider', { name: 'Replay time' });
        await page.waitForFunction(() => Number(document.querySelector('input[type=range]')?.value) > 1);
        const advanced = Number(await slider.inputValue());
        await page.getByRole('button', { name: 'Pause', exact: true }).click();
        await slider.evaluate(el => {
          Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, el.max);
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await page.waitForSelector('[data-end-card]');
        const result = await page.locator('[data-end-card]').innerText();
        assert(result.includes(n === 1 ? 'The robot is out.' : n === 2 ? 'Both robots are out.' : 'All three robots are out.'));
        await page.screenshot({ path: path.join(OUT, key + '.png') });
        checks.push({ key, available, check: '3D frame drawn; clock advanced; scrubbed to result', advanced, result });
      }
    }
    assert.deepEqual(errors, []);
    const result = { browser: browser.version(), url: URL, liveTransport: 'controlled refusal; no AWS tasks launched', checks, refusals, errors };
    fs.writeFileSync(path.join(OUT, 'browser_backend.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(`PASS ${checks.length} setup checks; ${checks.filter(c => c.check.startsWith('3D')).length} replay playback checks; no console or page errors`);
  } finally { await context.close(); await browser.close(); }
})().catch(e => { console.error(e.stack); process.exit(1); });

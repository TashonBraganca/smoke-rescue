const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require('/private/tmp/ryor-browser/node_modules/playwright');
const OUT = path.resolve(__dirname, '../../handoff/ryor_v2_evidence');
const URL = process.env.RYOR_URL || 'http://host.docker.internal:8202/rescue/';
(async () => {
  const browser = await chromium.connect(process.env.BROWSER_WS);
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  const page = await context.newPage();
  const errors = [], missionRequests = [];
  page.on('request', r => { if (r.url().includes('/api/mission')) missionRequests.push(r.url()); });
  page.on('pageerror', e => { errors.push(e.stack); console.log('PAGE ERROR', e.stack); });
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text());  });
  try {
    await page.goto(URL);
    await page.waitForSelector('[data-house]', { timeout: 15000 });
    assert.equal(await page.locator('[data-house]').count(), 5);
    await page.waitForFunction(() => document.querySelector('[data-start]')?.textContent === 'Start rescue');
    await page.screenshot({ path: path.join(OUT, 'picker_desktop.png'), fullPage: true });
    assert.equal(await page.locator('iframe').count(), 0);
    assert.equal(await page.locator('select').count(), 0);
    await page.getByRole('button', { name: '2 robots', exact: true }).click();
    const start = Date.now();
    await page.locator('[data-start]').click();
    await page.waitForFunction(() => Number(document.querySelector('[data-first-frame]')?.getAttribute('data-first-frame')) > 0);
    const firstFrameMs = Date.now() - start;
    assert(firstFrameMs < 2000, `First frame took ${firstFrameMs} ms`);
    await page.screenshot({ path: path.join(OUT, 'replay_start.png') });
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    const seek = async t => page.locator('input[type=range]').evaluate((el, value) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(el, String(value));
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }, t);
    await seek(90);
    await page.getByRole('button', { name: 'Play', exact: true }).click();
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(OUT, 'replay_middle.png') });
    const middle = await page.locator('[data-caption]').innerText();
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await page.getByRole('button', { name: 'Full house', exact: true }).click();
    await page.screenshot({ path: path.join(OUT, 'replay_full_house.png') });
    await seek(168);
    await page.getByRole('button', { name: 'Play', exact: true }).click();
    await page.waitForSelector('[data-end-card]', { timeout: 30000 });
    await page.waitForTimeout(700);
    await page.screenshot({ path: path.join(OUT, 'replay_end.png'), fullPage: true });
    const end = await page.locator('[data-end-card]').innerText();
    assert(end.includes('Both robots are out'));
    assert(end.includes('1 robot'));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(OUT, 'phone_end.png'), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await seek(90);
    await page.screenshot({ path: path.join(OUT, 'phone_replay.png') });
    await page.getByRole('button', { name: 'Choose another house', exact: true }).click();
    await page.screenshot({ path: path.join(OUT, 'phone_picker.png'), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.setViewportSize({ width: 1440, height: 1050 });
    // Every available combination must open a real, validated EASY replay.
    const setups = [];
    for (let house = 0; house < 5; house++) {
      for (let robots = 1; robots <= 3; robots++) {
        await page.locator('[data-house]').nth(house).click();
        await page.getByRole('button', { name: `${robots} robot${robots > 1 ? 's' : ''}`, exact: true }).click();
        await page.waitForFunction(() => document.querySelector('[data-start]')?.textContent === 'Start rescue');
        await page.locator('[data-start]').click();
        await page.waitForFunction(() => document.querySelector('[data-setup]'));
        const key = await page.locator('[data-setup]').getAttribute('data-setup');
        assert(key.endsWith(`__r${robots}__easy`));
        setups.push(key); console.log('Opened', key);
        await seek(180);
        await page.waitForSelector('[data-end-card]');
        const audit = JSON.parse(fs.readFileSync(path.join(OUT, 'recordings_audit.json')))[key];
        if (!audit.score.all_out) assert((await page.locator('[data-end-card]').innerText()).includes('returned after entering'));
        if (key === 'hx_035feb0125__r3__easy') await page.screenshot({ path: path.join(OUT, 'partial_return_result.png'), fullPage: true });
        await page.getByRole('button', { name: 'Choose another house', exact: true }).click();
      }
    }
    assert.equal(new Set(setups).size, 15);
    fs.writeFileSync(path.join(OUT, 'browser_errors.json'), JSON.stringify(errors, null, 2)); assert.deepEqual(errors, []); assert.deepEqual(missionRequests, [], 'Playback must not wait for a mission API');
    const report = { browser: browser.version(), url: URL, firstFrameMs, middle, end, setups, missionRequests, errors };
    fs.writeFileSync(path.join(OUT, 'browser_acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { await context.close(); await browser.close(); }
})().catch(e => { console.error(e.stack); process.exit(1); });

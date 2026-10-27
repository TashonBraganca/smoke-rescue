const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require('/private/tmp/ryor-browser/node_modules/playwright');
const OUT = path.resolve(__dirname, '../../handoff/ryor_v2_evidence');
(async () => {
  const b = await chromium.connect(process.env.BROWSER_WS);
  const c = await b.newContext({ viewport: { width: 1440, height: 1050 }, reducedMotion: 'reduce' });
  const p = await c.newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  p.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  try {
    // Exercise the actual main-site navigation, not a hand-written test wrapper.
    await p.goto('http://host.docker.internal:8202/', { waitUntil: 'domcontentloaded' });
    await p.getByRole('navigation', { name: 'Main', exact: true }).getByRole('link', { name: 'Try it', exact: true }).click();
    await p.waitForSelector('[data-house]');
    assert(p.url().endsWith('/rescue/'));
    await p.waitForFunction(() => document.querySelector('[data-start]')?.textContent === 'Start rescue');
    // Keyboard selection must work and keep its selected state.
    const bungalow = p.locator('[data-house]').nth(3);
    await bungalow.focus(); await p.keyboard.press('Enter');
    assert.equal(await bungalow.getAttribute('aria-pressed'), 'true');
    await p.getByRole('button', { name: '3 robots', exact: true }).click();
    await p.waitForFunction(() => document.querySelector('[data-start]')?.textContent === 'Start rescue');
    await p.locator('[data-start]').click();
    await p.waitForSelector('[data-setup="hx_035feb0125__r3__easy"]');
    assert(await p.getByRole('button', { name: 'Follow robots', exact: true }).isVisible());
    await p.getByRole('button', { name: 'Pause', exact: true }).click();
    await p.setViewportSize({ width: 320, height: 740 });
    await p.screenshot({ path: path.join(OUT, 'phone_320_reduced_motion.png'), fullPage: true });
    assert(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await p.getByRole('button', { name: 'Choose another house', exact: true }).click();
    await p.setViewportSize({ width: 768, height: 1024 });
    await p.screenshot({ path: path.join(OUT, 'tablet_picker.png'), fullPage: true });
    assert(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await p.emulateMedia({ reducedMotion: 'no-preference' });
    await p.setViewportSize({ width: 390, height: 844 });
    await p.locator('[data-house]').first().click();
    await p.getByRole('button', { name: '2 robots', exact: true }).click();
    await p.waitForFunction(() => document.querySelector('[data-start]')?.textContent === 'Start rescue');
    await p.locator('[data-start]').click();
    await p.getByRole('button', { name: 'Pause', exact: true }).click();
    await p.locator('input[type=range]').evaluate(el => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '90');
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await p.waitForTimeout(700);
    await p.screenshot({ path: path.join(OUT, 'phone_follow.png') });
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(OUT, 'browser_integration.json'), JSON.stringify({ browser: b.version(), navRoute: p.url(), keyboard: true, reducedMotion: true, widths: [320, 390, 768], overflow: false, errors }, null, 2));
    console.log('PASS native navigation, keyboard selection, reduced motion, 320 px and 768 px, no console errors');
  } finally { await c.close(); await b.close(); }
})().catch(e => { console.error(e.stack); process.exit(1); });

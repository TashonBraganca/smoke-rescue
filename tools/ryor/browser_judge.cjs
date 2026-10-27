// Headless checks use the production export. Mocked transport cases are labelled separately.
const fs = require('fs'), path = require('path'), assert = require('assert/strict');
const { chromium } = require('/private/tmp/ryor-browser/node_modules/playwright');
const ROOT = path.resolve(__dirname, '../..'), OUT = path.join(ROOT, 'handoff/judge_flow_evidence');
const URL = process.env.RYOR_URL || 'http://host.docker.internal:8202';
(async () => {
  const browser = await chromium.connect(process.env.BROWSER_WS);
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  const page = await context.newPage(), errors = [], missions = [];
  page.on('pageerror', e => errors.push(e.stack));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', r => { if (r.url().includes('/api/mission')) missions.push(r.url()); });
  const screenshot = name => page.screenshot({ path: path.join(OUT, name), fullPage: true });
  const result = { transportMocks: false, sizes: [], errors, missions };
  try {
    console.log('Real picker checks');
    await page.goto(URL + '/rescue/');
    await page.waitForSelector('[data-house]');
    assert.equal(await page.locator('[data-house]').count(), 5);
    assert.equal(await page.getByRole('button', { name: '2 robots', exact: true }).count(), 0);
    await screenshot('picker_desktop.png');
    for (const width of [320, 390, 768]) {
      await page.setViewportSize({ width, height: 844 });
      await screenshot(`picker_${width}.png`);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      result.sizes.push(width);
    }
    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.locator('[data-house]').nth(3).click();
    assert.equal(await page.getByRole('button', { name: '2 robots', exact: true }).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: '3 robots', exact: true }).click();
    await screenshot('picker_live.png');
    await page.locator('[data-house]').first().click();
    assert.equal(await page.getByRole('button', { name: '3 robots', exact: true }).count(), 0);
    const clicked = Date.now();
    await page.locator('[data-start]').click();
    await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
    result.introReadyMs = Date.now() - clicked;
    await page.waitForFunction(() => document.querySelector('video')?.currentTime > 0.3);
    await screenshot('cinematic_intro.png');
    await page.waitForSelector('[data-film-phase="mission"]', { timeout: 25000 });
    await screenshot('film_pending.png');
    assert.equal(missions.length, 0, 'Filmed houses must never launch an AWS run');
    await page.getByRole('button', { name: 'Try another house', exact: true }).first().click();
    await page.locator('[data-house]').nth(3).click();
    await page.locator('[data-start]').click();
    await page.getByRole('button', { name: 'Skip cinematic intro', exact: true }).click();
    await page.waitForSelector('[role=alert]');
    await screenshot('unchecked_replay_withheld.png');
    assert((await page.locator('section[aria-label="Live rescue"] [role=alert]').innerText()).includes('checked replay'));
    console.log('Landing checks');
    await page.goto(URL + '/');
    await page.waitForSelector('#mission [data-house]');
    assert.equal(await page.locator('#mission [data-house]').count(), 5);
    assert.equal(await page.getByRole('link', { name: 'Choose a mission', exact: true }).getAttribute('href'), '#mission');
    await page.locator('#mission').scrollIntoViewIfNeeded();
    await screenshot('landing_picker.png');
    await context.close();
    console.log('Film transition fixture checks');
    const mocked = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    const p = await mocked.newPage();
    p.on('pageerror', e => errors.push(e.stack));
    const config = JSON.parse(fs.readFileSync('/Volumes/Tashon/opencv/showcase/site-next/public/rescue-data/missions.json'));
    // Test-only metadata and short clip exercise transitions. This is not a house mission result.
    config.films.family_ranch = { src: config.intro, peopleFound: 4, peopleTotal: 5, missionSeconds: 123, allOut: true, note: 'Browser transition fixture' };
    await p.route('**/rescue-data/missions.json', r => r.fulfill({ json: config }));
    await p.goto(URL + '/rescue/?house=family_ranch&robots=3&start=1');
    for (const phase of ['intro','mission','outro']) {
      await p.waitForSelector(`[data-film-phase="${phase}"]`);
      await p.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
      await p.locator('video').evaluate(v => { v.currentTime = v.duration - 0.25; void v.play(); });
    }
    await p.waitForSelector('[data-end-card]');
    assert((await p.locator('[data-end-card]').innerText()).includes('4 / 5'));
    await p.screenshot({ path: path.join(OUT, 'film_end_transition_fixture.png'), fullPage: true });
    await p.setViewportSize({ width: 390, height: 844 });
    await p.screenshot({ path: path.join(OUT, 'film_end_phone_transition_fixture.png'), fullPage: true });
    assert(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    result.filmTransitionFixture = 'intro -> mission -> outro -> end, forced near clip ends; test-only metadata';
    await p.getByRole('link', { name: 'Run a live simulation on AWS', exact: true }).click();
    await p.waitForSelector('[data-house]');
    assert.equal(await p.locator('[data-house][aria-pressed=true]').getAttribute('data-house'), 'hx_46bfe067cc');
    assert.equal(await p.getByRole('button', { name: '2 robots', exact: true }).getAttribute('aria-pressed'), 'true');
    await mocked.close();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(OUT, 'browser_judge.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });

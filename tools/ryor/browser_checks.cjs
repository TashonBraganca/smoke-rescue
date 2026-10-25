/* Real public-page checks. Connect to an isolated headless browser; no API responses are mocked. */
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/private/tmp/ryor-browser/node_modules/playwright');
const { execFileSync } = require('child_process');
const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'handoff/ryor_evidence');
const URL = 'https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws/rescue/';
const mode = process.argv[2] || 'gallery';
const aws = (...args) => JSON.parse(execFileSync('aws', ['--region', 'us-east-1', ...args], {encoding:'utf8'}));
const save = (name, value) => fs.writeFileSync(path.join(OUT, name), JSON.stringify(value, null, 2));
const log = (...args) => console.log(new Date().toISOString(), ...args);

(async () => {
  const browser = await chromium.connect(process.env.BROWSER_WS);
  const context = await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,colorScheme:'dark'});
  const page = await context.newPage();
  const errors = [], requests = [], polls = [];
  let mission = null, killed = false;
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', async response => {
    if (!response.url().includes('/api/mission')) return;
    try {
      const body = await response.json();
      requests.push({url:response.url(),status:response.status(),at:Date.now(),body});
      if (response.request().method() === 'POST' && response.url().endsWith('/api/mission') && body.source === 'live') mission = body.mission;
      if (body.status) polls.push(body);
    } catch (_) {}
  });
  try {
    log('browser',browser.version(),'mode',mode);
    await page.goto(URL + '#house=family_ranch&robots=2&difficulty=medium');
    await page.waitForSelector('.card');
    if (await page.locator('.card').count() !== 27) throw Error('Gallery must contain all 27 houses');
    await page.screenshot({path:path.join(OUT,'public_gallery.png'),fullPage:true});
    const desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    await page.click('#edit-scene');
    await page.fill('#place-x','-3'); await page.fill('#place-y','3'); await page.click('#place-point');
    const rejection = await page.textContent('#placement-message');
    if (!rejection.includes('inside a room')) throw Error('Wall placement did not show its rejection');
    await page.screenshot({path:path.join(OUT,'public_placement_rejected.png')});
    for (const p of [[1.75,3.5],[5.25,3],[-2.25,8.25]]) {
      await page.fill('#place-x',String(p[0])); await page.fill('#place-y',String(p[1])); await page.click('#place-point');
    }
    await page.click('[data-tool="fire"]');
    await page.fill('#place-x','-.25'); await page.fill('#place-y','2.5'); await page.click('#place-point');
    const placement = await page.textContent('#placement-message');
    if (!placement.includes('3 of 5') || !placement.includes('Fire origin placed')) throw Error('Placement controls lost a point');
    await page.screenshot({path:path.join(OUT,'public_placement.png')});
    await page.click('#close-editor');
    if (mode === 'gallery') {
      await page.setViewportSize({width:390,height:844});
      await page.screenshot({path:path.join(OUT,'public_phone_gallery.png'),fullPage:true});
      const phoneOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      save('browser_gallery.json',{browser:browser.version(),url:URL,cards:27,rejection,placement,desktopOverflow,phoneOverflow,errors});
      if (desktopOverflow || phoneOverflow || errors.length) throw Error('Gallery browser check failed');
      log('PASS gallery, placement, rejection, phone, no page errors');
      return;
    }
    if (mode === 'custom') {
      await page.click('#edit-scene'); await page.click('#prepare-scene');
      await page.waitForFunction(() => !document.getElementById('watch-prepared').hidden || document.getElementById('placement-message').classList.contains('bad'),null,{timeout:700000});
      if (await page.locator('#watch-prepared').isHidden()) throw Error(await page.textContent('#placement-message'));
      await page.screenshot({path:path.join(OUT,'public_custom_prepared.png')});
      await page.click('#close-editor');
    } else {
      await page.click('#random-scene');
    }
    const start = Date.now();
    await page.click('#start');
    await page.screenshot({path:path.join(OUT,`public_${mode}_starting.png`)});
    let first = null, runningShot = false, finished = false;
    for (let i=0;i<350;i++) { // 700 s exceeds the API's 600 s wall limit plus browser fallback/retry time.
      await page.waitForTimeout(2000);
      const state = await page.evaluate(() => ({badge:document.getElementById('source-badge').textContent,
        clock:document.getElementById('clock').textContent,ready:!document.getElementById('scrubber').disabled,
        max:+document.getElementById('scrubber').max,score:!document.getElementById('score').hidden,
        error:!document.getElementById('wait-error').hidden,text:document.getElementById('sim-state').textContent,
        note:document.getElementById('recorded-note').textContent}));
      if (state.error) throw Error(await page.textContent('#wait-error-text'));
      if (state.ready && first === null) { first=(Date.now()-start)/1000; log('first playable frame',first,'s',mission); }
      if (state.ready && !runningShot && state.clock !== '0:00 / 3:00') {
        await page.screenshot({path:path.join(OUT,`public_${mode}_running.png`)});runningShot=true;
      }
      if (mode === 'kill' && mission && !killed && polls.some(r=>r.status.state==='running' && r.status.chunks>=2)) {
        const record = aws('dynamodb','get-item','--table-name','smoke-rescue-ryor','--key',JSON.stringify({pk:{S:'m#'+mission}})).Item;
        aws('ecs','stop-task','--cluster','smoke-rescue','--task',record.task.S,'--reason','ryor public browser fallback test');
        killed=true; log('stopped live task',mission);
      }
      if (mode === 'kill' && killed && state.badge.includes('Recorded') && state.ready) {
        await page.screenshot({path:path.join(OUT,'public_killed_fallback.png')});
        log('recorded fallback visible',state.note); finished=true; break;
      }
      if (state.score) {
        await page.screenshot({path:path.join(OUT,`public_${mode}_finished.png`)});
        finished=true; log('finished',((Date.now()-start)/1000),'s');break;
      }
      if (i%15===0) log(state.badge,state.clock,state.text);
    }
    if (!finished) throw Error('Mission browser check timed out');
    if (mode === 'kill' && !killed) throw Error('Kill test never stopped a live task');
    if (mode !== 'kill' && requests.some(r=>r.body.source==='recorded')) throw Error('Expected a live mission, received fallback');
    const compact = requests.map(r=>({url:r.url,status:r.status,at:r.at,body:r.body.status?{status:r.body.status,chunks:r.body.chunks.map(c=>({chunk:c.chunk,frames:c.frames.length}))}:r.body}));
    save(`browser_${mode}.json`,{browser:browser.version(),url:URL,mission,killed,first_playable_s:first,total_s:(Date.now()-start)/1000,errors,requests:compact});
    save(`browser_${mode}_replay.json`,polls);
    if (errors.length) throw Error('Page errors: '+errors.join('; '));
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(OUT,`public_phone_${mode}.png`),fullPage:true});
    if (await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw Error('Phone replay overflows horizontally');
    log('PASS',mode);
  } finally {
    save(`browser_${mode}_errors.json`,errors);
    await context.close(); await browser.close();
  }
})().catch(e=>{console.error(e.stack);process.exit(1)});

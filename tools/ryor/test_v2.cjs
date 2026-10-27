const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert/strict');
const SITE = process.env.RYOR_SITE || '/Volumes/Tashon/opencv/showcase/site-next';
const ts = require(path.join(SITE, 'node_modules/typescript'));
const code = ts.transpileModule(fs.readFileSync(path.join(SITE, 'src/rescue/data.ts'), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const moduleData = { exports: {} };
vm.runInNewContext(code, { exports: moduleData.exports, module: moduleData, fetch: () => { throw Error('No network in core checks'); }, Map });
const data = moduleData.exports;
assert.equal(data.advanceClock(0, -0.016, 4), 0, 'An older animation timestamp must not move the clock below zero');
assert.equal(data.advanceClock(179.9, 0.1, 4), 180);
assert.equal(data.advanceClock(20, 10, 4), 20.4, 'Returning from a hidden tab must not skip the mission');
const gallery = JSON.parse(fs.readFileSync(path.join(SITE, 'public/rescue-data/gallery.json')));
assert.equal(gallery.difficulty, 'easy');
assert.equal(gallery.houses.length, 5);
const counts = [];
for (const h of gallery.houses) {
  if (h.source === 'HouseExpo') assert(h.rooms >= 6 && h.rooms <= 9);
  assert(h.people === 4 || h.people === 5);
  for (let n = 1; n <= 3; n++) {
    const key = data.keyFor(h, n);
    const run = JSON.parse(fs.readFileSync(path.join(SITE, 'public/rescue-data', key + '.json')));
    assert.equal(run.setup, key);
    assert.equal(run.difficulty, 'easy');
    assert(run.validated);
    assert.equal(run.frames.length, 361);
    if (key === 'family_ranch__r2__easy') assert.equal(data.caption(h, run, 33.5), 'Robot A found someone in the bedroom.', 'Use the confirmed person location, not the robot location');
    assert.equal(run.score.robots_out, run.frames.at(-1).robots.filter(r => r.out).length);
    assert.equal(h.scores[n].rooms_searched, run.rooms_searched);
    for (const f of run.frames) {
      assert.equal(f.robots.length, n);
      for (const p of f.robots) assert([p.x, p.y, p.yaw].every(Number.isFinite));
      for (const marker of f.found) assert.equal(h.people_truth.filter(p => Math.hypot(p.x - marker.x, p.y - marker.y) <= 1.5).length, 1, 'A lit figure must match exactly one real person');
      const text = data.caption(h, run, f.t);
      assert(!/[\u2013\u2014]/.test(text));
      assert(!/go_to|explore:|mission_time|robot_/.test(text));
    }
    // The presentation must not turn a partial return into an all-out caption.
    const end = data.caption(h, run, 180);
    if (run.score.robots_out < n) assert(end.includes('returned after entering'), 'A robot that never entered is not necessarily still inside');
    counts.push(key);
  }
}
console.log(`PASS clock boundaries, all ${counts.length} EASY recordings, captions, partial exits and comparison scores`);

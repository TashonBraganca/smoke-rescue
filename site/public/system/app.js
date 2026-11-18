(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const vision = new Set(['assess_visibility', 'scan_thermal', 'inspect_candidate', 'track_fire_spread', 'measure_opening']);
  let steps = [], visible = 30, revision = 0;
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function json(value) { return JSON.stringify(value, null, 2); }
  async function request(url, options) {
    const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(60000), ...options});
    const text = await response.text();
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
    return text;
  }
  async function health() {
    $('health').textContent = 'API checking';
    $('refresh-health').disabled = true;
    try {
      const data = JSON.parse(await request('/api/health'));
      if (data.status !== 'ok') throw new Error('Service did not report ok');
      $('health').textContent = `API healthy / OpenCV ${data.opencv || 'unknown'} / ${data.backend || 'unknown'} / ${data.model || 'model unspecified'} / approvals ${data.approval_mode || 'unknown'}`;
      $('health').dataset.state = 'ok';
    } catch (error) {
      $('health').textContent = 'API unavailable / recorded results still available';
      $('health').dataset.state = 'error';
      $('health').title = error.message;
    } finally { $('refresh-health').disabled = false; }
  }
  function groupTrace(text) {
    const groups = new Map();
    let invalid = 0;
    text.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
        if (!event || typeof event !== 'object' || Array.isArray(event)) throw new Error('Invalid record');
      } catch (_) { invalid++; return; }
      const key = event.step || `${event.kind || 'record'} at line ${i + 1}`;
      if (!groups.has(key)) groups.set(key, {id: key, records: [], actors: new Set()});
      const step = groups.get(key);
      step.records.push(event);
      if (event.robot) step.actors.add(event.robot);
      if (event.step) step.actors.add(event.step.split('@')[0]);
      (event.orders || []).forEach(order => { if (order.robot) step.actors.add(order.robot); });
      if (event.arguments && event.arguments.robot) step.actors.add(event.arguments.robot);
    });
    return {steps: Array.from(groups.values()), invalid};
  }
  function field(parent, title, value, empty) {
    parent.append(node('h4', title));
    if (value == null || (Array.isArray(value) && !value.length)) parent.append(node('p', empty || 'Not recorded.', 'muted'));
    else parent.append(typeof value === 'string' ? node('p', value) : node('pre', json(value)));
  }
  function renderStep(step) {
    const card = node('article', undefined, 'step');
    const start = step.records.find(e => e.kind === 'step_start');
    const end = step.records.find(e => e.kind === 'step_end');
    const actor = start?.robot || step.id.split('@')[0];
    card.dataset.actor = actor;
    const head = node('div', undefined, 'step-head');
    head.append(node('h3', step.id), node('span', end?.latency_s != null ? `${end.latency_s} s total latency` : 'Total latency not recorded'));
    card.append(head);
    const approval = step.records.find(e => e.kind === 'approval' || e.kind === 'approval_answer');
    field(card, 'Event', start?.event || (approval ? approval : null), actor === 'commander' ? 'Commander coordination. Trigger not recorded.' : 'Trigger not recorded.');
    card.append(node('h4', 'Executed tools'));
    const calls = step.records.filter(e => e.kind === 'tool');
    if (!calls.length) card.append(node('p', 'No executed tool calls recorded.', 'muted'));
    calls.forEach(call => {
      const row = node('div', undefined, 'tool-call');
      row.append(node('strong', `${call.name} / ${vision.has(call.name) ? 'OpenCV 5' : 'action or control'} / round ${call.round ?? '?'}`));
      row.append(node('p', `Result ID: ${call.result_id ?? 'not recorded'} | Verdict: ${call.verdict ?? 'not recorded'}`));
      if (call.error) row.append(node('p', `Tool error: ${call.error}`, 'error'));
      if (call.refused) row.append(node('p', `Refused: ${call.reason || 'reason not recorded'}`, 'error'));
      if (call.ok === false) row.append(node('p', `Failed: ${call.reason || 'reason not recorded'}`, 'error'));
      if (call.arguments && Object.keys(call.arguments).length) row.append(node('pre', json(call.arguments)));
      card.append(row);
    });
    step.records.filter(e => e.kind === 'model_error').forEach(e => card.append(node('p', `Model error: ${e.error}`, 'error')));
    field(card, 'Orders', end?.orders, end ? 'No orders.' : 'Step incomplete. Orders not recorded.');
    field(card, 'Refusals', end?.refusals, end ? 'No refusals.' : 'Step incomplete. Refusals not recorded.');
    field(card, 'Model summary', end?.summary, 'Summary not recorded.');
    const raw = node('details');
    raw.append(node('summary', 'Raw records, model replies and per-call latency'), node('pre', json(step.records)));
    card.append(raw);
    return card;
  }
  function renderTrace() {
    const actor = $('robot').value, tool = $('tool-filter').value;
    const filtered = steps.filter(step => (!actor || step.actors.has(actor)) && (!tool || step.records.some(e => e.kind === 'tool' && e.name === tool)));
    $('steps').replaceChildren(...filtered.slice(0, visible).map(renderStep));
    $('trace-status').textContent = `${filtered.length} matching decision steps. Showing ${Math.min(visible, filtered.length)}.`;
    $('more').hidden = filtered.length <= visible;
  }
  function options(id, values, label) {
    $(id).replaceChildren(new Option(label, ''), ...Array.from(values).sort().map(value => new Option(value, value)));
  }
  function renderMetrics(rows) {
    if (!Array.isArray(rows) || !rows.length || rows.some(row => !row || typeof row.recording !== 'string')) throw new Error('Expected the nonempty array written by brain.compare_runs.');
    const table = node('table');
    table.append(node('caption', 'Recorded metrics. Missing values are shown as not recorded.'));
    const thead = node('thead'), tr = node('tr');
    const metricHead = node('th', 'Metric'); metricHead.scope = 'col'; tr.append(metricHead);
    rows.forEach(row => {
      const th = node('th'); th.scope = 'col';
      th.append(node('span', row.agent && Object.keys(row.agent).length ? 'With agents' : 'Without agents'), node('small', row.recording));
      tr.append(th);
    });
    thead.append(tr); table.append(thead);
    const body = node('tbody');
    const metrics = [
      ['Duration, s', 'duration_s'], ['Survivors in ground truth', 'survivors_total'],
      ['Real survivors found', 'survivors_found'], ['First real survivor, s', 'first_found_s'],
      ['Confirmations', 'confirmations'], ['False confirmations', 'false_confirmations'],
      ['Search area proxy, m²', 'area_searched_m2'], ['Area searched by both, m²', 'area_searched_by_both_m2'],
      ['Double coverage share', 'double_coverage_share'], ['Heat exposure by robot', 'heat'],
      ['Collision time, s by robot', 'collision_s'], ['Agent steps', 'agent.steps'],
      ['Agent orders', 'agent.orders'], ['Agent refusals', 'agent.refusals'], ['Agent errors in run summary', 'agent.errors'],
      ['Agent orders by type', 'agent.orders_by_type']
    ];
    metrics.forEach(([label, path]) => {
      const row = node('tr'), th = node('th', label); th.scope = 'row'; row.append(th);
      rows.forEach(data => {
        const value = path.split('.').reduce((a, key) => a == null ? undefined : a[key], data);
        const cell = node('td');
        if (value == null) cell.textContent = path === 'first_found_s' && value === null ? 'None found' : 'Not recorded';
        else if (typeof value === 'object') cell.append(node('pre', json(value)));
        else cell.textContent = String(value);
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body); $('metrics-table').replaceChildren(table);
    $('metrics-status').textContent = `${rows.length} recordings loaded. Agent labels follow the presence of agent metadata.`;
  }
  async function loadRun() {
    const version = ++revision;
    const base = `runs/${encodeURIComponent($('run').value)}/`;
    steps = []; $('steps').replaceChildren(); $('metrics-table').replaceChildren(); $('more').hidden = true;
    options('robot', [], 'All actors'); options('tool-filter', [], 'All tools');
    $('trace-status').textContent = 'Loading decisions.'; $('metrics-status').textContent = 'Loading comparison.';
    $('video-status').textContent = 'Loading recorded walkthrough.';
    ['video', 'trace', 'compare'].forEach((name, i) => {
      const link = $(`${name}-link`); link.href = base + ['walkthrough.mp4', 'agent_trace.jsonl', 'compare.json'][i]; link.hidden = false;
    });
    $('video').src = base + 'walkthrough.mp4'; $('video').load();
    $('run-status').textContent = `Recorded run / ${$('run').value}`;
    await Promise.all([
      (async () => {
        try {
          const text = await request(base + 'agent_trace.jsonl');
          if (version !== revision) return;
          const parsed = groupTrace(text); steps = parsed.steps;
          options('robot', new Set(steps.flatMap(s => Array.from(s.actors))), 'All actors');
          options('tool-filter', new Set(steps.flatMap(s => s.records.filter(e => e.kind === 'tool').map(e => e.name)).filter(Boolean)), 'All tools');
          visible = 30; renderTrace();
          if (parsed.invalid) $('trace-status').textContent += ` Skipped ${parsed.invalid} malformed lines.`;
        } catch (error) { if (version === revision) $('trace-status').textContent = `Trace unavailable. ${error.message}`; }
      })(),
      (async () => {
        try {
          const text = await request(base + 'compare.json');
          if (version === revision) renderMetrics(JSON.parse(text));
        } catch (error) { if (version === revision) $('metrics-status').textContent = `Comparison unavailable. ${error.message}`; }
      })()
    ]);
  }
  async function catalog() {
    try {
      const data = JSON.parse(await request('runs/index.json'));
      if (!Array.isArray(data.runs) || data.runs.some(r => !r || typeof r.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/.test(r.id))) throw new Error('Invalid runs/index.json catalog.');
      if (!data.runs.length) throw new Error('No runs published yet. See runs/README.txt.');
      $('run').replaceChildren(...data.runs.map(r => new Option(r.label || r.id, r.id)));
      $('run').disabled = false;
      await loadRun();
    } catch (error) {
      $('run').replaceChildren(new Option('No runs available', ''));
      $('run-status').textContent = `Cannot load runs. ${error.message}`;
      $('trace-status').textContent = 'No recorded trace loaded.';
      $('metrics-status').textContent = 'No comparison loaded.';
    }
  }
  $('refresh-health').addEventListener('click', health);
  $('run').addEventListener('change', loadRun);
  ['robot', 'tool-filter'].forEach(id => $(id).addEventListener('change', () => { visible = 30; renderTrace(); }));
  $('more').addEventListener('click', () => { visible += 30; renderTrace(); });
  $('video').addEventListener('loadedmetadata', () => { $('video-status').textContent = 'Recorded walkthrough ready.'; });
  $('video').addEventListener('error', () => { $('video-status').textContent = 'Video unavailable or unsupported. Try the download link.'; });
  health(); catalog();
})();

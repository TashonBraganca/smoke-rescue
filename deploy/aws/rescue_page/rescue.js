/* Run your own rescue: gallery, setup, live mission polling and canvas replay.
   Data and HTTP contract: tools/ryor/CONTRACT.txt. Same-origin: the API lives at /api/mission on this host. */
(function () {
  "use strict";

  var API = document.currentScript.getAttribute('data-api') || "/api/mission";
  var GALLERY_URL = "gallery.json";
  var MISSION_S = 180; // The bounded engine mission, in simulated seconds.
  var DT = 0.5; // Match the inclusive replay frame cadence.
  var CELL = 0.1, ORIGIN = -25, GRID_W = 500; // The brain's fixed gate-frame map grid.
  var POLL_MS = 2000; // Keep idle browser polling below one request per second.
  var LIMIT_STARTING_S = 150;   // still "starting" after this: give up and show the recorded run
  var LIMIT_STALL_S = 150;      // running but no new data for this long
  var LIMIT_TOTAL_S = 660;      // contract hard limit is 600 s wall on the server
  var MAX_NET_FAILS = 6; // Retry temporary network failures before loading the recording.
  var editedHouse = null, roomLayout = null, layoutValid = true;
  var scene = null, preparedSceneKey = null, preparing = false, placementTool = 'person', prepGeneration = 0, readySetups = null;
  if (/[?&]devlimits=1/.test(location.search)) { LIMIT_STARTING_S = 15; LIMIT_STALL_S = 15; LIMIT_TOTAL_S = 60; } // dev server tests only
  var IDS = ["robot_A", "robot_B", "robot_C"];
  var LETTER = { robot_A: "A", robot_B: "B", robot_C: "C" };
  var STATE_WORDS = { explore: "searching", search: "searching", launch: "at the door", wait: "at the door",
    enter: "going in", "return": "heading out", exit: "heading out", out: "outside", home: "heading out",
    stuck: "stuck", recover: "getting unstuck", rescue: "with a person", complete: "outside", returning: "heading out" };
  var REASONS = {
    busy: "All live slots on AWS are in use right now.",
    rate_limited: "Several runs were started from your network in a short time, so this one was not started live.",
    daily_limited: 'The daily live-mission budget has been used.',
    disabled: "Live runs are switched off at the moment.",
    network: "The live service did not answer.",
    stuck: "The cloud machine took too long to start.",
    stalled: "The live run stopped sending data.",
    timeout: "The live run went past its time limit."
  };
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function fmt(t) { t = Math.max(0, Math.round(t)); return Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0"); }
  function letter(id) { return LETTER[id] || String(id || "?").replace(/^robot_/, ""); }
  function tagEl(id) { var i = el("i", "tag " + letter(id).toLowerCase(), letter(id)); return i; }
  function num(v) { return typeof v === "number" && isFinite(v); }

  /* ---------- theme ---------- */
  var themeOrder = ["auto", "light", "dark"];
  var theme = "auto";
  try { theme = localStorage.getItem("ryor-theme") || "auto"; } catch (e) {}
  var colors = {};
  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    ["--paper", "--bg", "--ink", "--muted", "--line", "--grid", "--wall", "--room", "--room-alt", "--robot-a", "--robot-b",
      "--robot-c", "--fire", "--fire-core", "--person", "--accent"].forEach(function (k) { colors[k.slice(2)] = cs.getPropertyValue(k).trim(); });
    colors.robot = { robot_A: colors["robot-a"], robot_B: colors["robot-b"], robot_C: colors["robot-c"] };
  }
  function applyTheme() {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    $("theme").textContent = "Theme: " + theme;
    $("theme").setAttribute("aria-label", "Colour theme: " + theme + ". Press to change.");
    readColors();
    redrawAll();
  }
  $("theme").addEventListener("click", function () {
    theme = themeOrder[(themeOrder.indexOf(theme) + 1) % 3];
    try { localStorage.setItem("ryor-theme", theme); } catch (e) {}
    applyTheme();
  });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { readColors(); redrawAll(); });

  /* ---------- plan drawing (shared by thumbnails, preview, replay) ---------- */
  function viewBox(h) {
    var b = h.bounds || [-5, 0, 5, 10];
    var x0 = b[0], y0 = b[1], x1 = b[2], y1 = b[3];
    (h.pads || []).concat(h.door ? [h.door] : []).forEach(function (p) {
      x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    });
    var m = 0.8;
    return [x0 - m, y0 - m, x1 + m, y1 + m];
  }
  function makeXf(vb, w, h, pad) {
    var s = Math.min((w - 2 * pad) / (vb[2] - vb[0]), (h - 2 * pad) / (vb[3] - vb[1]));
    var ox = (w - s * (vb[2] - vb[0])) / 2, oy = (h - s * (vb[3] - vb[1])) / 2;
    return { s: s, x: function (x) { return ox + (x - vb[0]) * s; }, y: function (y) { return oy + (vb[3] - y) * s; } };
  }
  function wallBounds(h) {
    var b = [1e9, 1e9, -1e9, -1e9];
    (h.walls || []).forEach(function (w) { b[0] = Math.min(b[0], w[0], w[2]); b[1] = Math.min(b[1], w[1], w[3]); b[2] = Math.max(b[2], w[0], w[2]); b[3] = Math.max(b[3], w[1], w[3]); });
    return b[0] < b[2] ? b : (h.bounds || [-5, 0, 5, 10]);
  }
  function drawFurniture(ctx, h, xf) {
    ctx.fillStyle = colors.line;
    (h.furniture || []).forEach(function (poly) { if (Array.isArray(poly) && poly.length > 2) { polyPath(ctx, xf, poly); ctx.globalAlpha = 0.55; ctx.fill(); ctx.globalAlpha = 1; } });
  }
  function drawLowWalls(ctx, h, xf) {
    if (!h.low_walls || !h.low_walls.length) return;
    ctx.save(); ctx.strokeStyle = colors.wall; ctx.globalAlpha = 0.45; ctx.lineWidth = Math.max(1, Math.min(2, xf.s * 0.05)); ctx.setLineDash([3, 3]);
    ctx.beginPath(); h.low_walls.forEach(function (w) { ctx.moveTo(xf.x(w[0]), xf.y(w[1])); ctx.lineTo(xf.x(w[2]), xf.y(w[3])); }); ctx.stroke(); ctx.restore();
  }
  function polyPath(ctx, xf, poly) {
    ctx.beginPath();
    poly.forEach(function (p, i) { if (i) ctx.lineTo(xf.x(p[0]), xf.y(p[1])); else ctx.moveTo(xf.x(p[0]), xf.y(p[1])); });
    ctx.closePath();
  }
  function drawPlan(ctx, h, xf, opt) {
    opt = opt || {};
    (h.rooms_poly || []).forEach(function (r, i) {
      if (!r.poly || r.poly.length < 3) return;
      polyPath(ctx, xf, r.poly);
      ctx.fillStyle = i % 2 ? colors["room-alt"] : colors.room;
      ctx.fill();
    });
    drawFurniture(ctx, h, xf);
    if (opt.labels && xf.s > 14) {
      ctx.fillStyle = colors.muted;
      ctx.font = "500 " + Math.max(10, Math.min(12, xf.s * 0.32)) + "px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      (h.rooms_poly || []).forEach(function (r) {
        if (!r.poly || !r.kind) return;
        var cx = 0, cy = 0; r.poly.forEach(function (p) { cx += p[0]; cy += p[1]; });
        cx /= r.poly.length; cy /= r.poly.length;
        ctx.globalAlpha = 0.7; ctx.fillText(r.kind, xf.x(cx), xf.y(cy)); ctx.globalAlpha = 1;
      });
    }
    ctx.strokeStyle = colors.wall;
    ctx.lineWidth = Math.max(1.2, Math.min(3.5, xf.s * 0.1));
    ctx.lineCap = "square";
    ctx.beginPath();
    (h.walls || []).forEach(function (w) { ctx.moveTo(xf.x(w[0]), xf.y(w[1])); ctx.lineTo(xf.x(w[2]), xf.y(w[3])); });
    ctx.stroke();
    drawLowWalls(ctx, h, xf);
    if (h.door) {
      ctx.fillStyle = colors.accent;
      var d = Math.max(3, xf.s * 0.18);
      ctx.beginPath(); ctx.moveTo(xf.x(h.door[0]), xf.y(h.door[1]) - d); ctx.lineTo(xf.x(h.door[0]) - d, xf.y(h.door[1]) + d);
      ctx.lineTo(xf.x(h.door[0]) + d, xf.y(h.door[1]) + d); ctx.closePath(); ctx.fill();
    }
  }
  function drawPads(ctx, h, xf, n) {
    (h.pads || []).slice(0, n).forEach(function (p, i) {
      var id = IDS[i], r = Math.max(5, xf.s * 0.26);
      ctx.strokeStyle = colors.robot[id]; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(xf.x(p[0]), xf.y(p[1]), r, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    });
  }
  function sizeCanvas(c, cssW, cssH) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2.5); // Bound pixel work while preserving readable phone plans.
    // Responsive plan canvases keep CSS sizing. Fixed inline widths otherwise survive a desktop-to-phone resize.
    if (c.id === 'replay') { c.style.width = cssW + "px"; c.style.height = cssH + "px"; }
    c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr);
    var ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /* ---------- gallery and setup ---------- */
  var G = null, houses = [], sel = { house: null, robots: 2, difficulty: "medium" }, filter = "all";
  var DIFF_FALLBACK = { easy: { head_start_s: 0, visibility_m: 6.0 }, medium: { head_start_s: 90, visibility_m: 4.0 }, hard: { head_start_s: 180, visibility_m: 2.5 } };
  function houseById(id) { if (editedHouse && editedHouse.id === id) return editedHouse; for (var i = 0; i < houses.length; i++) if (houses[i].id === id) return houses[i]; return null; }
  function diffInfo(d) { return (G && G.difficulties && G.difficulties[d]) || DIFF_FALLBACK[d] || {}; }
  function diffSentence(d) {
    var i = diffInfo(d), hs = i.head_start_s, vis = i.visibility_m;
    if (i.label) return i.label;
    var fire = !num(hs) ? "" : hs === 0 ? "The fire starts as the robots go in" : "The fire has a " + (hs >= 120 ? (hs / 60) + " minute" : hs + " s") + " head start";
    var smoke = !num(vis) ? "" : vis >= 6 ? "light smoke, people visible from up to 6 m" : "smoke limits sight to " + vis + " m";
    return fire && smoke ? fire + "; " + smoke + "." : (i.label || "");
  }
  function sourceLabel(h) { return h.source === "HouseExpo" ? "HouseExpo" : "ours"; }

  function renderGallery() {
    var ul = $("gallery"); ul.textContent = "";
    var shown = houses.filter(function (h) { return filter === "all" || (filter === "ours" ? h.source !== "HouseExpo" : h.source === "HouseExpo"); });
    $("gallery-count").textContent = shown.length + " of " + houses.length + " houses";
    shown.forEach(function (h) {
      var li = el("li"), b = el("button", "card"); b.type = "button";
      b.setAttribute("aria-pressed", String(sel.house === h.id));
      b.dataset.id = h.id;
      var c = el("canvas"); c.setAttribute("aria-hidden", "true");
      b.appendChild(c);
      b.appendChild(el("span", "src " + (h.source === "HouseExpo" ? "hx" : "ours"), sourceLabel(h)));
      b.appendChild(el("span", "name", h.name || h.id));
      var meta = el("span", "meta");
      meta.appendChild(el("span", null, (h.rooms || "?") + " rooms"));
      meta.appendChild(el("span", null, (num(h.floor_m2) ? Math.round(h.floor_m2) : "?") + " m²"));
      meta.appendChild(el("span", null, (h.people || "?") + " people"));
      b.appendChild(meta);
      b.setAttribute("aria-label", (h.name || h.id) + ", " + sourceLabel(h) + ", " + h.rooms + " rooms, " + Math.round(h.floor_m2 || 0) + " square metres");
      b.addEventListener("click", function () { pickHouse(h.id, true); });
      li.appendChild(b); ul.appendChild(li);
    });
    drawThumbs();
  }
  function drawThumbs() {
    Array.prototype.forEach.call(document.querySelectorAll(".card"), function (b) {
      var h = houseById(b.dataset.id), c = b.querySelector("canvas");
      if (!h || !c) return;
      var w = c.clientWidth || 170, hh = c.clientHeight || 128;
      var ctx = sizeCanvas(c, w, hh);
      ctx.clearRect(0, 0, w, hh);
      var b2 = wallBounds(h);
      var xf = makeXf([b2[0] - .4, b2[1] - .4, b2[2] + .4, b2[3] + .4], w, hh, 4);
      drawPlan(ctx, h, xf, {});
    });
  }
  function drawPreview() {
    var c = $("preview"), w = c.clientWidth || 340, hh = Math.round(w * 260 / 400);
    var ctx = sizeCanvas(c, w, hh);
    ctx.clearRect(0, 0, w, hh);
    var h = houseById(sel.house);
    if (!h) return;
    var xf = makeXf(viewBox(h), w, hh, 8);
    drawPlan(ctx, h, xf, { labels: true });
    drawPads(ctx, h, xf, sel.robots);
    if (scene) drawPlacementMarks(ctx, xf, scene);
  }
  function pickHouse(id, scroll) {
    var h = houseById(id); if (!h) return;
    sel.house = id;
    resetScene();
    Array.prototype.forEach.call(document.querySelectorAll(".card"), function (b) { b.setAttribute("aria-pressed", String(b.dataset.id === id)); });
    var cap = $("preview-cap"); cap.textContent = "";
    var nm = el("b", null, h.name || h.id); cap.appendChild(nm);
    var txt = ": " + h.rooms + " rooms, " + Math.round(h.floor_m2) + " m², " + h.people + " people inside. ";
    txt += h.source === "HouseExpo" ? "Synthetic residential layout from HouseExpo" + (h.source_id ? " (" + h.source_id + ")" : "") + "." : "One of our own test houses.";
    cap.appendChild(document.createTextNode(txt));
    $("start").disabled = false;
    drawPreview();
    saveHash();
    updateStartAvailability();
    if (scroll && window.innerWidth <= 980) $("setup").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
  function setSeg(groupId, attr, val) {
    Array.prototype.forEach.call($(groupId).querySelectorAll("button"), function (b) { b.setAttribute("aria-pressed", String(b.dataset[attr] === String(val))); });
  }
  $("robots").addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b) return;
    invalidateScene();
    sel.robots = +b.dataset.n; setSeg("robots", "n", sel.robots); drawPreview(); saveHash();
    updateStartAvailability();
  });
  $("difficulty").addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b) return;
    invalidateScene();
    sel.difficulty = b.dataset.d; setSeg("difficulty", "d", sel.difficulty); renderDiffLine(); saveHash();
    updateStartAvailability();
  });
  function renderDiffLine() {
    var line = $("diff-line"); line.textContent = "";
    line.appendChild(el("b", null, sel.difficulty[0].toUpperCase() + sel.difficulty.slice(1) + ". "));
    line.appendChild(document.createTextNode(diffSentence(sel.difficulty)));
  }
  document.querySelector(".filters").addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b) return;
    filter = b.dataset.filter;
    Array.prototype.forEach.call(this.querySelectorAll("button"), function (x) { x.setAttribute("aria-pressed", String(x === b)); });
    renderGallery();
  });
  function saveHash() {
    var q = "house=" + encodeURIComponent(sel.house || "") + "&robots=" + sel.robots + "&difficulty=" + sel.difficulty;
    try { history.replaceState(null, "", "#" + q); } catch (e) {}
  }
  function readHash() {
    var q = {}; location.hash.replace(/^#/, "").split("&").forEach(function (kv) { var p = kv.split("="); if (p[0]) q[p[0]] = decodeURIComponent(p[1] || ""); });
    if (q.robots && /^[123]$/.test(q.robots)) sel.robots = +q.robots;
    if (q.difficulty && DIFF_FALLBACK[q.difficulty]) sel.difficulty = q.difficulty;
    return q.house || null;
  }

  function setCitation(text) {
    var p = $("citation"); p.textContent = "";
    p.appendChild(el("strong", null, "Real floor plans. "));
    var m = String(text).match(/https?:\/\/\S+?(?=[,\s]|$)/);
    if (!m) { p.appendChild(document.createTextNode(text)); return; }
    var i = text.indexOf(m[0]);
    p.appendChild(document.createTextNode(text.slice(0, i)));
    var a = el("a", null, m[0].replace(/^https?:\/\//, "")); a.href = m[0]; p.appendChild(a);
    p.appendChild(document.createTextNode(text.slice(i + m[0].length) + " People and fire are placed by our scenario generator."));
  }
  window.addEventListener("hashchange", function () {
    if (!houses.length || !$("mission").hidden) return;
    var want = readHash();
    setSeg("robots", "n", sel.robots); setSeg("difficulty", "d", sel.difficulty); renderDiffLine();
    if (houseById(want)) pickHouse(want, false); else drawPreview();
  });
  function loadGallery() {
    fetchJSON(GALLERY_URL, 20000).then(function (g) {
      if (!g || !Array.isArray(g.houses) || !g.houses.length) throw new Error("empty gallery");
      G = g; MISSION_S = num(g.mission_s) ? g.mission_s : 180;
      houses = g.houses.filter(function (h) { return h && h.id && Array.isArray(h.walls) && Array.isArray(h.bounds); });
      houses.sort(function (a, b) { return (a.source === "HouseExpo") - (b.source === "HouseExpo") || (a.floor_m2 || 0) - (b.floor_m2 || 0); });
      if (!houses.some(function (h) { return h.source === "HouseExpo"; })) document.querySelector(".filters").hidden = true;
      // The static attribution includes the full licence link and the SUNCG-derived origin.
      var want = readHash();
      setSeg("robots", "n", sel.robots); setSeg("difficulty", "d", sel.difficulty); renderDiffLine();
      renderGallery();
      pickHouse(houseById(want) ? want : houses[0].id, false);
      var shared = new URL(location.href).searchParams.get('mission');
      if (shared && /^[a-z0-9_:]{8,120}$/.test(shared)) fetchJSON(API + '/' + shared + '?from=19', 20000).then(function (reply) {
        var status = reply.status, parts = status.setup.split('__'), h = houseById(parts[0]);
        if (!h || M) return;
        pickHouse(h.id, false); sel.robots = +parts[1].slice(1); sel.difficulty = parts[2];
        setSeg('robots', 'n', sel.robots); setSeg('difficulty', 'd', sel.difficulty); renderDiffLine();
        scene = status.placement || null; preparedSceneKey = status.setup;
        roomLayout = status.layout || null; editedHouse = status.house || null; if (editedHouse) h = editedHouse;
        $('terra').checked = !!status.terra;
        startMission(h, +parts[1].slice(1), parts[2], false, shared, status.state === 'done');
      }).catch(function (error) { $('gallery-error').hidden = false; $('gallery-error').textContent = 'This replay could not be loaded. ' + error.message; });
    }).catch(function (err) {
      var p = $("gallery-error"); p.hidden = false;
      p.textContent = "The list of houses did not load (" + err.message + "). Reload the page to try again.";
    });
    refreshHealth();
  }
  function refreshHealth() {
    fetchJSON(API + "/health", 8000).then(function (h) {
      if (!h) return;
      if (Array.isArray(h.ready_setups)) { readySetups = new Set(h.ready_setups); updateStartAvailability(); }
      var p = $("health"); p.hidden = false;
      p.textContent = h.ok ? "Live service is up: " + h.running + " of " + h.cap + " cloud machines in use." : "Live service reports a problem; you will get recorded runs.";
      if (h.cache_setups < h.cache_expected) setTimeout(refreshHealth, 30000); // Let newly prepared setups become selectable without reloading.
    }).catch(function () {});
  }
  function updateStartAvailability() {
    if (!sel.house) return;
    var key = sel.house + '__r' + sel.robots + '__' + sel.difficulty;
    var missing = !scene && readySetups && !readySetups.has(key);
    $('start').disabled = !!missing || !layoutValid;
    $('start').textContent = missing ? 'Recording is preparing' : scene && !preparedSceneKey ? 'Prepare my scene' : 'Start the rescue';
    if (missing) $('scene-summary').textContent = 'The recording for this setup is still preparing. Choose another robot count or house, or place your own scene.';
    else if (!scene) $('scene-summary').textContent = "People and fire follow this house's fixed generator seed.";
  }

  /* ---------- network ---------- */
  function fetchJSON(url, ms, opts) {
    opts = opts || {};
    var ctl = window.AbortController ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, ms || 15000) : null;
    if (ctl) opts.signal = ctl.signal;
    opts.cache = "no-store";
    return fetch(url, opts).then(function (r) {
      if (timer) clearTimeout(timer);
      if (!r.ok) return r.json().catch(function () { return {}; }).then(function (body) {
        var e = new Error(body.error || ("HTTP " + r.status)); e.status = r.status; e.body = body; throw e;
      });
      return r.json();
    }, function (e) { if (timer) clearTimeout(timer); throw new Error(e && e.name === "AbortError" ? "timed out" : "network error"); });
  }

  /* Placement is world-side configuration. It is never sent to a robot or to Terra. */
  function placementMessage(text, bad) { $('placement-message').textContent = text; $('placement-message').className = bad ? 'bad' : ''; }
  function invalidateScene() {
    preparedSceneKey = null; prepGeneration++; preparing = false;
    if (scene) $('start').textContent = 'Prepare my scene';
    $('prepare-scene').disabled = false; $('watch-prepared').hidden = true;
  }
  function resetScene() {
    editedHouse = null; roomLayout = null; layoutValid = true;
    scene = null; invalidateScene();
    $('start').textContent = 'Start the rescue';
    $('scene-summary').textContent = "People and fire follow this house's fixed generator seed.";
    updateStartAvailability();
  }
  function openEditor() {
    if (!scene) { scene = { people: [], fire: null }; invalidateScene(); $('scene-summary').textContent = 'Custom scene. Place people and one fire origin, or choose Random to restore the generator.'; }
    updateStartAvailability();
    if (!$('scene-editor').open) $('scene-editor').showModal();
    drawEditor();
  }
  $('edit-scene').addEventListener('click', openEditor);
  $('close-editor').addEventListener('click', function () { $('scene-editor').close(); drawPreview(); });
  $('random-scene').addEventListener('click', function () { resetScene(); drawPreview(); });
  $('clear-scene').addEventListener('click', function () { scene = { people: [], fire: null }; invalidateScene(); drawEditor(); placementMessage('Place up to five people and one fire origin.'); });
  Array.prototype.forEach.call(document.querySelectorAll('[data-tool]'), function (button) {
    button.addEventListener('click', function () {
      placementTool = button.dataset.tool;
      Array.prototype.forEach.call(document.querySelectorAll('[data-tool]'), function (b) { b.setAttribute('aria-pressed', String(b === button)); });
    });
  });
  function drawPlacementMarks(ctx, xf, value) {
    ctx.font = '700 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    value.people.forEach(function (p, i) {
      var x = xf.x(p[0]), y = xf.y(p[1]);
      ctx.fillStyle = colors.person; ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = colors.paper; ctx.fillText(String(i + 1), x, y);
    });
    if (value.fire) {
      ctx.fillStyle = colors.fire; ctx.beginPath(); ctx.arc(xf.x(value.fire[0]), xf.y(value.fire[1]), 11, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = colors.ink; ctx.fillText('Fire', xf.x(value.fire[0]), xf.y(value.fire[1]) - 19);
    }
  }
  function drawEditor() {
    var h = houseById(sel.house), c = $('placement'); if (!h || !c || !scene) return;
    var w = c.clientWidth || 700, height = c.clientHeight || 420, ctx = sizeCanvas(c, w, height);
    ctx.clearRect(0, 0, w, height);
    var transform = makeXf(viewBox(h), w, height, 20);
    drawPlan(ctx, h, transform, { labels: true }); drawPads(ctx, h, transform, sel.robots);
    drawPlacementMarks(ctx, transform, scene);
  }
  function pointInRect(p, r, inset) { return p[0] >= r[0] + inset && p[0] <= r[2] - inset && p[1] >= r[1] + inset && p[1] <= r[3] - inset; }
  function tryPlacement(p) {
    if (preparing) return placementMessage('This scene is preparing. Close the editor to keep waiting, or clear it to make another scene.', true);
    // Match the server's 0.25 m fire grid, 0.6 m wall margin, 0.9 m furniture envelope and 1.2 m door margin.
    p = p.map(function (v) { return Math.round(v * 4) / 4; });
    if (!p.every(num)) return placementMessage('Enter two finite coordinates.', true);
    var h = houseById(sel.house), geometry = h.placement_geometry;
    if (placementTool === 'erase') {
      var nearest = -1, distance = 1; // A one-metre touch target works on small phone plans.
      scene.people.forEach(function (q, i) { var d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d < distance) { nearest = i; distance = d; } });
      if (nearest >= 0) scene.people.splice(nearest, 1);
      else if (scene.fire && Math.hypot(p[0] - scene.fire[0], p[1] - scene.fire[1]) < 1) scene.fire = null;
      else return placementMessage('Tap close to a placed person or the fire to remove it.');
    } else {
      var room = geometry.rooms.find(function (r) { return pointInRect(p, r.rect, .6); });
      if (!room) return placementMessage('Place this inside a room, at least 0.6 m from its walls.', true);
      var blocked = geometry.furniture.some(function (f) {
        var size = geometry.footprints.furniture[f.kind].size_m;
        return pointInRect(p, [f.at[0] - size[0] / 2, f.at[1] - size[1] / 2, f.at[0] + size[0] / 2, f.at[1] + size[1] / 2], -.9);
      });
      if (blocked) return placementMessage('That point is too close to furniture. Choose clear floor.', true);
      if (geometry.doors.some(function (d) { return Math.hypot(p[0] - d.at[0], p[1] - d.at[1]) < 1.2; })) return placementMessage('Keep 1.2 m clear around every doorway.', true);
      if (scene.people.some(function (q) { return Math.hypot(p[0] - q[0], p[1] - q[1]) < 1.2; })) return placementMessage('Keep people and fire at least 1.2 m apart.', true);
      if (placementTool === 'person') {
        if (scene.people.length >= 5) return placementMessage('Five people are already placed. Remove one to move them.', true);
        if (scene.fire && Math.hypot(p[0] - scene.fire[0], p[1] - scene.fire[1]) < 1.2) return placementMessage('Keep a person at least 1.2 m from the fire.', true);
        scene.people.push(p);
      } else {
        if (geometry.combustible_rooms.indexOf(room.id) < 0) return placementMessage('The fire needs combustible floor. Choose a bedroom or living room.', true);
        scene.fire = p;
      }
    }
    invalidateScene(); drawEditor(); drawPreview();
    $('scene-summary').textContent = scene.people.length + ' people placed. ' + (scene.fire ? 'Fire placed. Prepare this scene before starting.' : 'Place the fire next.');
    placementMessage(scene.people.length + ' of 5 people placed. ' + (scene.fire ? 'Fire origin placed. Ready to validate and prepare.' : 'Choose Fire origin and place it on clear floor.'));
  }
  $('place-point').addEventListener('click', function () { tryPlacement([Number($('place-x').value), Number($('place-y').value)]); });
  $('placement').addEventListener('click', function (event) {
    var c = $('placement'), rect = c.getBoundingClientRect(), h = houseById(sel.house), vb = viewBox(h);
    var transform = makeXf(vb, rect.width, rect.height, 20);
    var x = (event.clientX - rect.left - transform.x(0)) / transform.s;
    var y = (transform.y(0) - (event.clientY - rect.top)) / transform.s;
    $('place-x').value = Math.round(x * 4) / 4; $('place-y').value = Math.round(y * 4) / 4;
    tryPlacement([x, y]);
  });
  $('prepare-scene').addEventListener('click', function () {
    if (!scene || !scene.fire) return placementMessage('Place one fire origin before preparing the scene.', true);
    var my = ++prepGeneration, started = performance.now(); preparing = true; $('prepare-scene').disabled = true;
    placementMessage('Checking the scene and starting its recorded fallback on AWS.');
    var body = { house: sel.house, robots: sel.robots, difficulty: sel.difficulty, placement: scene, layout: roomLayout, prepare: true, terra: false };
    fetchJSON(API, 30000, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(function (result) {
      if (my !== prepGeneration) return;
      function ready() {
        preparing = false; preparedSceneKey = result.setup; $('prepare-scene').disabled = false;
        $('start').textContent = 'Start the rescue'; $('watch-prepared').hidden = false;
        $('scene-summary').textContent = scene.people.length + ' people placed. Exact recorded fallback ready.';
        placementMessage('Your scene is ready. Start a live mission, or watch the prepared brain-only run.');
      }
      function check() {
        if (my !== prepGeneration) return;
        fetchJSON(API + '/' + result.mission + '?from=19', 20000).then(function (reply) {
          if (my !== prepGeneration) return;
          if (reply.status.state === 'done') return ready();
          if (reply.status.state === 'failed') throw new Error('Preparation stopped early. Your scene is kept. Press Prepare to retry.');
          placementMessage('Preparing your exact scene, ' + Math.round(reply.status.sim_t || 0) + ' of 180 simulated seconds complete. ' + Math.round((performance.now() - started) / 1000) + ' seconds elapsed.');
          setTimeout(check, POLL_MS);
        }).catch(failed);
      }
      if (result.prepared) ready(); else check();
    }).catch(failed);
    function failed(error) { if (my !== prepGeneration) return; preparing = false; $('prepare-scene').disabled = false; placementMessage(error.message, true); }
  });
  $('watch-prepared').addEventListener('click', function () {
    if (!preparedSceneKey) return;
    $('scene-editor').close(); startMission(houseById(sel.house), sel.robots, sel.difficulty, true);
  });

  /* ---------- mission state ---------- */
  var M = null; // current mission
  var gen = 0;  // bumps on every new mission so stale polls are dropped

  function newMission(house, robots, difficulty) {
    return {
      house: house, robots: robots, difficulty: difficulty, key: house.id + "__r" + robots + "__" + difficulty,
      id: null, source: "live", reason: null, started: performance.now(), lastData: performance.now(),
      status: null, nextChunk: 0, frames: [], ready: -1, netFails: 0, done: false, failed: false,
      t: 0, playing: true, speed: 4, lastIdx: -1, seen: {}, union: null, unionCount: 0, perRobot: {},
      found: [], foundIds: {}, events: [], yawMap: null, moments: [], compare: null, scoreShown: false, errorShown: false,
      vb: viewBox(house), layers: {}, logCount: 0
    };
  }

  $("start").addEventListener("click", function () {
    if (!layoutValid) return;
    var h = houseById(sel.house); if (!h) return;
    if (scene && !preparedSceneKey) { openEditor(); return; }
    startMission(h, sel.robots, sel.difficulty);
  });
  $("back").addEventListener("click", function () { stopMission(); showChoose(); });
  $("try-more").addEventListener("click", function () { stopMission(); showChoose(); $("setup").scrollIntoView({ block: "start" }); });
  $("retry").addEventListener("click", function () { if (M) startMission(M.house, M.robots, M.difficulty); });
  $("replay-again").addEventListener("click", function () { if (!M) return; seek(0); M.playing = true; M.scoreShown = false; $("score").hidden = true; syncPlay(); });

  function showChoose() {
    $("mission").hidden = true; $("choose").hidden = false;
    document.title = "Run your own rescue | Smoke Rescue";
    var url = new URL(location.href); url.searchParams.delete('mission'); history.replaceState(null, '', url);
    drawThumbs(); drawPreview();
  }
  function stopMission() { gen++; M = null; }

  function startMission(h, robots, difficulty, recordedOnly, resumeId, saved) {
    gen++;
    var my = gen;
    if (scene) h = Object.assign({}, h, { people: scene.people.length, people_truth: scene.people.map(function (p, i) { return { id: 'p' + (i + 1), x: p[0], y: p[1] }; }) });
    M = newMission(h, robots, difficulty);
    M.scene = scene ? JSON.parse(JSON.stringify(scene)) : null;
    M.layout = roomLayout ? JSON.parse(JSON.stringify(roomLayout)) : null;
    M.key = preparedSceneKey || M.key;
    M.terra = $('terra').checked;
    $("choose").hidden = true; $("mission").hidden = false;
    window.scrollTo(0, 0);
    $("mission-h").textContent = h.name || h.id;
    $("mission-sub").textContent = robots + (robots === 1 ? " robot" : " robots") + ", " + difficulty + ". " + diffSentence(difficulty);
    document.title = (h.name || h.id) + " | Run your own rescue";
    setSource("live", null);
    $('mission-link').hidden = true;
    $("score").hidden = true; $("log").textContent = ""; $("log-empty").hidden = false; $("ticks").textContent = "";
    $("wait-error").hidden = true; $("waiting").hidden = false; $("stages").hidden = false; $("wait-clock").hidden = false;
    $("wait-h").textContent = "Getting the mission ready";
    $("wait-note").textContent = h.people + " people are somewhere in this house. The robots do not know where, or where the fire is. They start on the pads outside the door.";
    setStage("sent", "now");
    setSpeedButtons(4);
    renderTeam(); layoutStage(); draw();
    if (resumeId) {
      M.id = resumeId; updateMissionLink();
      if (resumeId.indexOf('cache:') === 0) setSource('recorded', 'This is a saved recording.');
      else if (saved) setSource('saved', 'This mission finished earlier. These are its recorded decisions and paths.');
      poll(my); fetchCompare(my); tick(); return;
    }
    if (recordedOnly) { M.id = 'cache:' + M.key; updateMissionLink(); setSource('recorded', 'This is the scene you prepared.'); poll(my); tick(); return; }
    fetchJSON(API, 30000, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ house: h.id, robots: robots, difficulty: difficulty, placement: M.scene, layout: M.layout, terra: M.terra }) })
      .then(function (r) {
        if (my !== gen) return;
        if (!r || !r.mission) throw new Error("no mission id");
        M.key = r.setup || M.key;
        M.id = r.mission; updateMissionLink();
        if (r.source === "recorded" || String(r.mission).indexOf("cache:") === 0) {
          M.id = r.mission; setSource("recorded", REASONS[r.reason] || ("The live run was not started (" + (r.reason || "no reason given") + ")."));
          setStage("sent", "done"); setStage("machine", "skip");
        } else {
          M.id = r.mission; setStage("sent", "done"); setStage("machine", "now");
        }
        poll(my);
      })
      .catch(function (e) {
        if (my !== gen) return;
        if (e.status === 400 || e.status === 409 || e.status === 429) return showError(e.message);
        fallback(my, "The live service did not answer (" + e.message + ").");
      });
    fetchCompare(my);
    tick();
  }

  function setSource(src, reason) {
    M.source = src; M.reason = reason;
    var b = $("source-badge"), note = $("recorded-note");
    if (src === "recorded") {
      b.className = "badge rec"; b.textContent = "Recorded run of this exact setup";
      note.hidden = false;
      note.textContent = (reason ? reason + " " : "") + "This recorded brain-only run uses the same house, people, fire, robot count and difficulty. Live Terra decisions can change the search path.";
    } else if (src === 'saved') {
      b.className = 'badge rec'; b.textContent = 'Saved live run'; note.hidden = false; note.textContent = reason;
    } else {
      b.className = "badge live"; b.textContent = "Live run on AWS"; note.hidden = true;
    }
  }
  function updateMissionLink() {
    var link = $('mission-link'); link.href = '?mission=' + encodeURIComponent(M.id); link.hidden = false;
  }

  function fallback(my, reason) {
    if (my !== gen || !M) return;
    if (M.source === "recorded") return showError("Neither the live run nor the recorded run could be loaded. " + reason);
    var keep = { house: M.house, robots: M.robots, difficulty: M.difficulty, key: M.key, scene: M.scene, layout: M.layout, compare: M.compare, t: M.ready >= 0 ? M.t : 0, speed: M.speed };
    gen++; my = gen;
    M = newMission(keep.house, keep.robots, keep.difficulty);
    M.key = keep.key; M.scene = keep.scene; M.layout = keep.layout;
    M.compare = keep.compare; M.resumeAt = keep.t; M.speed = keep.speed;
    M.id = "cache:" + M.key;
    updateMissionLink();
    $("log").textContent = ""; $("log-empty").hidden = false; $("ticks").textContent = ""; $("score").hidden = true;
    setSource("recorded", reason + (keep.t > 1 ? " The replay carries on from " + fmt(keep.t) + "." : ""));
    $("waiting").hidden = false; $("wait-h").textContent = "Loading the recorded run";
    setStage("sent", "done"); setStage("machine", "skip"); setStage("door", "now");
    announce("Switched to the recorded run of this exact setup.");
    renderTeam(); draw();
    poll(my);
  }

  function showError(msg) {
    if (!M) return;
    M.failed = true;
    $("recorded-note").hidden = true;
    $("waiting").hidden = false; $("wait-h").textContent = "This mission could not be shown";
    $("stages").hidden = true; $("wait-clock").hidden = true; $("wait-note").textContent = "";
    $("wait-error").hidden = false; $("wait-error-text").textContent = msg;
  }

  function poll(my) {
    if (my !== gen || !M || !M.id) return;
    var url = API + "/" + M.id + "?from=" + M.nextChunk;
    fetchJSON(url, 20000).then(function (r) {
      if (my !== gen) return;
      var st = r && r.status;
      if (!st) throw new Error("answer had no status");
      M.status = st;
      if (st.setup) M.key = st.setup;
      if (st.house && !M.layout) M.house = st.house;
      if (st.layout) M.layout = st.layout;
      if (st.placement && !M.scene) {
        M.scene = st.placement; M.house = Object.assign({}, M.house, { people: st.placement.people.length });
      }
      var got = ingest(r.chunks || []);
      if (got) M.lastData = performance.now();
      if (st.state === "failed") {
        if (M.source === "recorded") return showError("The recorded run is marked as failed (" + (st.error || "no reason") + ").");
        return fallback(my, "The live run stopped early" + (st.error ? " (" + plainError(st.error) + ")" : "") + ".");
      }
      if (st.source === "recorded" && M.source !== "recorded") setSource("recorded", st.fallback_reason || null);
      updateStages();
      M.netFails = 0; // only a fully handled answer counts as healthy
      var since = (performance.now() - M.lastData) / 1000, total = (performance.now() - M.started) / 1000;
      if (M.source === "live") {
        if (st.state === "starting" && total > LIMIT_STARTING_S) return fallback(my, REASONS.stuck);
        if (st.state === "running" && since > LIMIT_STALL_S) return fallback(my, REASONS.stalled);
        if (st.state !== "done" && total > LIMIT_TOTAL_S) return fallback(my, REASONS.timeout);
      }
      if (st.state === "done" && M.nextChunk >= (st.chunks || 0)) {
        M.done = true;
        if (M.ready < 0) {
          if (M.source === "live") return fallback(my, "The live run finished without any replay data.");
          return showError("The recorded run has no replay data.");
        }
        return;
      }
      var more = (st.chunks || 0) > M.nextChunk && got;
      setTimeout(function () { poll(my); }, more ? 120 : POLL_MS);
    }).catch(function (e) {
      if (my !== gen) return;
      if (!e.status && e.message !== "network error" && e.message !== "timed out" && window.console) console.error("poll", e);
      M.netFails++;
      if (e.status === 404 && M.source === "recorded") return showError("The recorded run of this setup is not available (" + e.message + ").");
      if (M.netFails >= MAX_NET_FAILS) return fallback(my, "The live service did not answer (" + e.message + ").");
      setTimeout(function () { poll(my); }, Math.min(8000, POLL_MS * M.netFails));
    });
  }
  function plainError(s) { return String(s).replace(/_/g, " ").replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"').replace(/\u2014/g, ',').replace(/\s*\(([^)]*)\)/g, ", $1").slice(0, 160); }

  function ingest(chunks) {
    var got = 0;
    chunks.sort(function (a, b) { return a.chunk - b.chunk; }).forEach(function (c) {
      if (!c || c.chunk !== M.nextChunk || !Array.isArray(c.frames)) return;
      c.frames.forEach(function (f) {
        if (!f || !num(f.t) || !Array.isArray(f.robots)) return;
        var i = Math.round(f.t / DT);
        if (i >= 0 && i <= MISSION_S / DT + 2) M.frames[i] = f;
      });
      M.nextChunk++; got++;
    });
    if (got) {
      var r = M.ready; while (M.frames[r + 1]) r++;
      M.ready = r;
      calibrateYaw();
      if (M.ready >= 0 && !$("waiting").hidden && !M.failed) {
        $("waiting").hidden = true; setStage("search", "done");
        $("play").disabled = false; $("scrubber").disabled = false;
        announce("The robots are inside. Replay started.");
      }
      buildTicks();
    }
    return got;
  }

  /* Robot yaw convention is not fixed by the contract. Fit it from motion: compare the direction each robot actually
     moves with the reported yaw under a few candidate conventions, pick the best one. */
  function calibrateYaw() {
    if (M.yawMap && M.yawMap.n > 80) return;
    var cands = [function (y) { return y; }, function (y) { return 90 - y; }, function (y) { return 90 + y; }, function (y) { return y - 90; }];
    var err = [0, 0, 0, 0], n = 0, prev = {};
    for (var i = 0; i <= M.ready; i++) {
      var f = M.frames[i]; if (!f) continue;
      f.robots.forEach(function (r) {
        var p = prev[r.id];
        if (p && num(r.yaw)) {
          var dx = r.x - p.x, dy = r.y - p.y;
          if (Math.hypot(dx, dy) > 0.1) {
            var mv = Math.atan2(dy, dx) * 180 / Math.PI;
            cands.forEach(function (fn, k) { var d = Math.abs(((fn(r.yaw) - mv) % 360 + 540) % 360 - 180); err[k] += Math.min(d, 90); });
            n++;
          }
        }
        prev[r.id] = r;
      });
    }
    if (n < 5) return;
    var best = 0; err.forEach(function (e, k) { if (e < err[best]) best = k; });
    M.yawMap = { fn: cands[best], n: n };
  }
  function headingRad(r) { var y = M.yawMap ? M.yawMap.fn(r.yaw) : r.yaw; return (num(y) ? y : 90) * Math.PI / 180; }

  /* ---------- stage list ---------- */
  function setStage(s, state) {
    var li = document.querySelector('.stages li[data-s="' + s + '"]'); if (!li) return;
    li.classList.remove("now", "done", "skip");
    if (state === "skip") { li.hidden = true; return; }
    li.hidden = false; if (state) li.classList.add(state);
  }
  function updateStages() {
    var st = M.status; if (!st) return;
    if (st.state === "starting") { setStage("machine", "now"); setStage("door", ""); }
    else if (st.state === "running" || st.state === "done") {
      if (M.source === "live") setStage("machine", "done");
      if (M.ready < 0) setStage("door", "now"); else { setStage("door", "done"); setStage("search", "done"); }
    }
  }

  /* ---------- comparison (cached 1-robot run) ---------- */
  function fetchCompare(my) {
    if (M.scene) return; // A different placement would be an invalid one-robot comparison.
    var other = M.robots === 1 ? 3 : 1;
    var key = M.house.id + "__r" + other + "__" + M.difficulty;
    fetchJSON(API + "/cache:" + key + "?from=999", 15000).then(function (r) {
      if (my !== gen || !M) return;
      var s = r && r.status && r.status.score;
      if (s) M.compare = { robots: other, score: s };
      if (M.scoreShown) renderScore();
    }).catch(function () {});
  }

  /* ---------- replay state ---------- */
  /* Cells inside the house's rooms. The robots' own maps also mark ground outside the walls; the page counts and
     draws only the floor inside the house, which is what "searched" means to a viewer. */
  function floorMask(h) {
    if (h._mask) return h._mask;
    var mask = new Uint8Array(GRID_W * GRID_W), n = 0;
    (h.rooms_poly || []).forEach(function (r) {
      var P = r.poly; if (!P || P.length < 3) return;
      var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      P.forEach(function (p) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); });
      var ix0 = Math.max(0, Math.floor((x0 - ORIGIN) / CELL)), ix1 = Math.min(GRID_W - 1, Math.floor((x1 - ORIGIN) / CELL));
      var iy0 = Math.max(0, Math.floor((y0 - ORIGIN) / CELL)), iy1 = Math.min(GRID_W - 1, Math.floor((y1 - ORIGIN) / CELL));
      for (var iy = iy0; iy <= iy1; iy++) {
        var cy = ORIGIN + (iy + 0.5) * CELL;
        for (var ix = ix0; ix <= ix1; ix++) {
          var cx = ORIGIN + (ix + 0.5) * CELL, inside = false;
          for (var a = 0, b = P.length - 1; a < P.length; b = a++) {
            if ((P[a][1] > cy) !== (P[b][1] > cy) && cx < (P[b][0] - P[a][0]) * (cy - P[a][1]) / (P[b][1] - P[a][1]) + P[a][0]) inside = !inside;
          }
          var k = iy * GRID_W + ix;
          if (inside && !mask[k]) { mask[k] = 1; n++; }
        }
      }
    });
    h._mask = n > 50 ? mask : null;
    h._floorCells = n > 50 ? n : (h.floor_m2 || 1) / (CELL * CELL);
    return h._mask;
  }
  function resetLayers() {
    M.seen = {}; M.perRobot = {}; M.found = []; M.foundIds = {}; M.events = []; M.moments = []; M.lastIdx = -1;
    M.mask = floorMask(M.house);
    M.union = new Uint8Array(GRID_W * GRID_W); M.unionCount = 0; M.logCount = 0; M.lastLink = null;
    var vb = M.vb;
    M.gx0 = Math.floor((vb[0] - ORIGIN) / CELL); M.gy0 = Math.floor((vb[1] - ORIGIN) / CELL);
    M.gw = Math.ceil((vb[2] - vb[0]) / CELL) + 1; M.gh = Math.ceil((vb[3] - vb[1]) / CELL) + 1;
    IDS.slice(0, M.robots).forEach(function (id) {
      var c = M.layers[id] || document.createElement("canvas");
      c.width = M.gw; c.height = M.gh;
      var ctx = c.getContext("2d"); ctx.clearRect(0, 0, M.gw, M.gh);
      M.layers[id] = c; M.perRobot[id] = 0;
    });
    $("log").textContent = ""; $("log-empty").hidden = false;
  }
  function applyFrame(i, live) {
    var f = M.frames[i]; if (!f) return;
    var seen = f.seen || {};
    Object.keys(seen).forEach(function (id) {
      var layer = M.layers[id]; if (!layer || !Array.isArray(seen[id])) return;
      var ctx = layer.getContext("2d"); ctx.fillStyle = colors.robot[id] || colors.muted;
      seen[id].forEach(function (cell) {
        if (M.mask && !M.mask[cell]) return;
        var ix = cell % GRID_W, iy = (cell - ix) / GRID_W;
        var px = ix - M.gx0, py = M.gh - 1 - (iy - M.gy0);
        if (px >= 0 && py >= 0 && px < M.gw && py < M.gh) ctx.fillRect(px, py, 1, 1);
        if (!M.union[cell]) { M.union[cell] = 1; M.unionCount++; }
        M.perRobot[id] = (M.perRobot[id] || 0) + 1;
      });
    });
    (f.found || []).forEach(function (p) {
      if (M.foundIds[p.id]) return;
      M.foundIds[p.id] = 1;
      M.found.push({ id: p.id, x: p.x, y: p.y, by: p.by, t: num(p.t) ? p.t : f.t, born: live ? performance.now() : 0 });
    });
    (f.events || []).forEach(function (ev) {
      var e = { t: num(ev.t) ? ev.t : f.t, robot: ev.robot, kind: ev.kind, text: ev.text || ev.kind };
      if (ev.kind === "map_shared") {
        var pl = M.lastLink;
        if (pl && pl.robot !== ev.robot && Math.abs(pl.t - e.t) < 1.5) {
          // the two robots each report the same link: show it once, naming both
          pl.text = "Robots " + letter(pl.robot) + " and " + letter(ev.robot) + " joined their maps into one shared map.";
          if (pl.li) pl.li.lastChild.textContent = pl.text;
          if (live) showMoment(pl);
          M.lastLink = null;
          return;
        }
      }
      M.events.push(e); e.li = addLog(e, live);
      if (ev.kind === "map_shared") { M.lastLink = e; M.moments.push(e); if (live) showMoment(e); return; }
      if (ev.kind === "mission_complete" || ev.kind === "robot_out") M.moments.push(e);
      if (live && (ev.kind === "survivor_confirmed" || ev.kind === "robot_out" || ev.kind === "mission_complete")) showMoment(e);
    });
    var prevF = M.frames[i - 1];
    f.robots.forEach(function (r) {
      var was = prevF && prevF.robots.filter(function (x) { return x.id === r.id; })[0];
      if (r.out && was && !was.out && !(f.events || []).some(function (e) { return e.robot === r.id && (e.kind === "robot_out" || e.kind === "out"); })) {
        var e = { t: f.t, robot: r.id, kind: "robot_out", text: "Robot " + letter(r.id) + " is back outside." };
        M.events.push(e); addLog(e, live); M.moments.push(e); if (live) showMoment(e);
      }
    });
  }
  function seek(t) {
    var target = Math.max(0, Math.min(t, M.ready * DT));
    var idx = Math.round(target / DT);
    if (idx < M.lastIdx || !M.union) resetLayers();
    var live = idx - M.lastIdx <= 4;
    for (var i = M.lastIdx + 1; i <= idx; i++) applyFrame(i, live);
    M.lastIdx = idx; M.t = target;
  }

  /* ---------- event log ---------- */
  var BIG = { map_shared: 1, survivor_confirmed: 1, survivor_from_peer: 0, room_marked_fire: 1, mission_complete: 1, robot_out: 1 };
  function addLog(e, live) {
    $("log-empty").hidden = true;
    var li = el("li", (BIG[e.kind] ? "big " : "") + (live && !reduceMotion ? "fresh" : ""));
    li.appendChild(el("time", null, fmt(e.t)));
    li.appendChild(e.robot ? tagEl(e.robot) : el("span"));
    li.appendChild(el("span", null, plainText(e)));
    var log = $("log"); log.insertBefore(li, log.firstChild);
    while (log.children.length > 120) log.removeChild(log.lastChild);
    if (live && BIG[e.kind]) announce(fmt(e.t) + ". " + plainText(e));
    return li;
  }
  function plainText(e) {
    var t = e.text || e.kind;
    if (e.kind === 'agent_order') t = t.replace(' applied an agent order.', ' reported an agent order.');
    return t.replace(/robot_([ABC])/g, "$1");
  }
  var srTimer = null;
  function announce(msg) { clearTimeout(srTimer); srTimer = setTimeout(function () { $("sr-live").textContent = msg; }, 60); }

  var momentTimer = null;
  function showMoment(e) {
    var m = $("moment"); m.textContent = "";
    var head = { map_shared: "Maps joined", survivor_confirmed: "Person found", robot_out: "Out of the house", mission_complete: "Heading out" }[e.kind] || "";
    if (e.robot) m.appendChild(tagEl(e.robot));
    m.appendChild(document.createTextNode(head + ". " + plainText(e)));
    m.hidden = false; m.style.opacity = "1";
    clearTimeout(momentTimer);
    momentTimer = setTimeout(function () { m.style.opacity = "0"; setTimeout(function () { m.hidden = true; }, 300); }, e.kind === "map_shared" ? 4200 : 2800);
  }

  function buildTicks() {
    var box = $("ticks"); box.textContent = "";
    for (var i = 0; i <= M.ready; i++) {
      var f = M.frames[i]; if (!f) continue;
      (f.events || []).forEach(function (ev) {
        var cls = ev.kind === "map_shared" ? "link" : ev.kind === "survivor_confirmed" ? "person" : (ev.kind === "robot_out" || ev.kind === "mission_complete") ? "out" : ev.kind === "room_marked_fire" ? "fire" : null;
        if (!cls) return;
        var t = el("i", cls); t.style.left = (100 * f.t / MISSION_S) + "%"; t.title = fmt(f.t) + " " + plainText(ev);
        box.appendChild(t);
      });
    }
  }

  /* ---------- team panel ---------- */
  function renderTeam() {
    var ul = $("team"); ul.textContent = "";
    IDS.slice(0, M.robots).forEach(function (id) {
      var li = el("li"); li.dataset.id = id;
      li.appendChild(tagEl(id));
      var w = el("span", "what"); w.appendChild(document.createTextNode("Robot " + letter(id))); w.appendChild(el("small", null, "waiting outside"));
      li.appendChild(w);
      var s = el("span", "stat"); s.innerHTML = "<b>0</b> m²"; li.appendChild(s);
      ul.appendChild(li);
    });
  }
  function updateTeam(f) {
    var floorCells = M.house._floorCells || (M.house.floor_m2 || 1) / (CELL * CELL);
    Array.prototype.forEach.call($("team").children, function (li) {
      var id = li.dataset.id, r = f && f.robots.filter(function (x) { return x.id === id; })[0];
      var found = M.found.filter(function (p) { return p.by === id; }).length;
      li.querySelector("small").textContent = r ? (r.out ? "outside" : (STATE_WORDS[r.state] || String(r.state || "").replace(/_/g, " "))) + (found ? ", found " + found : "") : "waiting outside";
      li.querySelector(".stat b").textContent = Math.round((M.perRobot[id] || 0) * CELL * CELL);
    });
    $("team-cov").textContent = Math.min(100, Math.round(100 * M.unionCount / floorCells)) + " % of floor searched";
    $("team-found").textContent = String(M.found.length);
  }

  /* ---------- canvas ---------- */
  var cv = $("replay"), cx = null, xf = null, cssW = 0, cssH = 0;
  function layoutStage() {
    if (!M) return;
    var stage = $("stage"), w = stage.clientWidth;
    var vb = M.vb, aspect = (vb[3] - vb[1]) / (vb[2] - vb[0]);
    var maxH = Math.max(280, Math.min(window.innerHeight * 0.68, 760));
    var h = Math.min(maxH, Math.max(260, w * aspect));
    cssW = w; cssH = Math.round(h);
    cx = sizeCanvas(cv, cssW, cssH);
    xf = makeXf(vb, cssW, cssH, 14);
  }
  window.addEventListener("resize", function () { if (M && !$("mission").hidden) { layoutStage(); draw(); } else { drawThumbs(); drawPreview(); } if ($('scene-editor').open) drawEditor(); });

  function frameAt() { return M && M.lastIdx >= 0 ? M.frames[M.lastIdx] : null; }
  function draw() {
    if (M && window.Rescue3D && window.Rescue3D.replayEnabled) window.Rescue3D.replay(M.house, frameAt(), M);
    if (!M || !cx) return;
    var ctx = cx;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = colors.paper; ctx.fillRect(0, 0, cssW, cssH);
    // 1 m grid, plan-sheet feel and an honest scale
    var vb = M.vb;
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (var gx = Math.ceil(vb[0]); gx <= vb[2]; gx++) { ctx.moveTo(Math.round(xf.x(gx)) + .5, 0); ctx.lineTo(Math.round(xf.x(gx)) + .5, cssH); }
    for (var gy = Math.ceil(vb[1]); gy <= vb[3]; gy++) { ctx.moveTo(0, Math.round(xf.y(gy)) + .5); ctx.lineTo(cssW, Math.round(xf.y(gy)) + .5); }
    ctx.stroke();
    var h = M.house;
    (h.rooms_poly || []).forEach(function (r, i) { if (r.poly && r.poly.length > 2) { polyPath(ctx, xf, r.poly); ctx.fillStyle = i % 2 ? colors["room-alt"] : colors.room; ctx.fill(); } });
    drawFurniture(ctx, h, xf);
    // searched floor per robot
    if (M.union) {
      ctx.save(); ctx.imageSmoothingEnabled = true; ctx.globalAlpha = 0.26;
      var px0 = xf.x(ORIGIN + M.gx0 * CELL), py0 = xf.y(ORIGIN + (M.gy0 + M.gh) * CELL);
      IDS.slice(0, M.robots).forEach(function (id) { if (M.layers[id]) ctx.drawImage(M.layers[id], px0, py0, M.gw * CELL * xf.s, M.gh * CELL * xf.s); });
      ctx.restore();
    }
    var f = frameAt();
    // fire
    var fire = null;
    for (var i = M.lastIdx; i >= 0 && !fire; i--) if (M.frames[i] && Array.isArray(M.frames[i].fire)) fire = M.frames[i].fire;
    (fire || []).forEach(function (fr) {
      if (!Array.isArray(fr) || !num(fr[0])) return;
      var x = xf.x(fr[0]), y = xf.y(fr[1]), r = Math.max(4, (fr[2] || 0.4) * xf.s);
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, colors["fire-core"]); g.addColorStop(0.55, colors.fire); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.globalAlpha = 0.85; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
    });
    // walls on top of tints and fire
    ctx.strokeStyle = colors.wall; ctx.lineWidth = Math.max(1.5, Math.min(4, xf.s * 0.1)); ctx.lineCap = "square";
    ctx.beginPath(); (h.walls || []).forEach(function (w) { ctx.moveTo(xf.x(w[0]), xf.y(w[1])); ctx.lineTo(xf.x(w[2]), xf.y(w[3])); }); ctx.stroke();
    drawLowWalls(ctx, h, xf);
    drawPlanMarks(ctx);
    // trails
    var trails = {};
    for (var j = 0; j <= M.lastIdx; j++) {
      var fj = M.frames[j]; if (!fj) continue;
      fj.robots.forEach(function (r) { if (num(r.x) && num(r.y)) (trails[r.id] = trails[r.id] || []).push(r); });
    }
    ctx.lineWidth = Math.max(1.4, xf.s * 0.05); ctx.lineJoin = "round";
    Object.keys(trails).forEach(function (id) {
      var pts = trails[id]; ctx.strokeStyle = colors.robot[id] || colors.muted; ctx.globalAlpha = 0.75; ctx.beginPath();
      pts.forEach(function (p, k) { if (k) ctx.lineTo(xf.x(p.x), xf.y(p.y)); else ctx.moveTo(xf.x(p.x), xf.y(p.y)); });
      ctx.stroke(); ctx.globalAlpha = 1;
    });
    // map link moment: dashed line between the two robots for 8 s of sim time
    if (f) M.moments.forEach(function (e) {
      if (e.kind !== "map_shared" || M.t - e.t > 8 || M.t < e.t) return;
      var a = f.robots.filter(function (r) { return r.id === e.robot; })[0]; if (!a) return;
      var b = null, bd = 1e9;
      f.robots.forEach(function (r) { if (r.id !== a.id) { var d = Math.hypot(r.x - a.x, r.y - a.y); if (d < bd) { bd = d; b = r; } } });
      if (!b) return;
      var k = 1 - (M.t - e.t) / 8;
      ctx.save(); ctx.globalAlpha = 0.35 + 0.65 * k; ctx.strokeStyle = colors.ink; ctx.lineWidth = 2.5; ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(xf.x(a.x), xf.y(a.y)); ctx.lineTo(xf.x(b.x), xf.y(b.y)); ctx.stroke(); ctx.setLineDash([]);
      var mx = xf.x((a.x + b.x) / 2), my = xf.y((a.y + b.y) / 2);
      ctx.fillStyle = colors.ink; ctx.font = "700 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText("maps joined", mx, my - 6); ctx.restore();
    });
    // Only confirmed people become replay markers, including at the final frame.
    var now = performance.now();
    M.found.forEach(function (p) {
      if (!num(p.x)) return;
      var age = p.born ? (now - p.born) / 500 : 1, s = reduceMotion ? 1 : Math.min(1, age);
      var r = Math.max(5, xf.s * 0.22);
      var x = xf.x(p.x), y = xf.y(p.y);
      if (!reduceMotion && age < 2) { ctx.strokeStyle = colors.robot[p.by] || colors.ink; ctx.globalAlpha = Math.max(0, 1 - age / 2); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r * (1 + age * 1.6), 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1; }
      ctx.fillStyle = colors.person; ctx.strokeStyle = colors.robot[p.by] || colors.ink; ctx.lineWidth = Math.max(2.5, r * 0.45);
      ctx.beginPath(); ctx.arc(x, y, r * (0.4 + 0.6 * s), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = colors.ink; ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      if (xf.s > 12) { var lt = xf.s > 34 ? "found by " + letter(p.by) + ", " + fmt(p.t) : fmt(p.t); var lw = ctx.measureText(lt).width; if (x + r + 4 + lw > cssW - 4) { ctx.textAlign = "right"; ctx.fillText(lt, x - r - 4, y); } else ctx.fillText(lt, x + r + 4, y); }
    });
    // robots
    var robots = f ? f.robots : (h.pads || []).slice(0, M.robots).map(function (p, i) { return { id: IDS[i], x: p[0], y: p[1], yaw: null, out: false, idle: true }; });
    robots.forEach(function (r) {
      if (!num(r.x) || !num(r.y)) return;
      var x = xf.x(r.x), y = xf.y(r.y), rad = Math.max(7, xf.s * 0.3), col = colors.robot[r.id] || colors.ink;
      var hd = r.idle ? Math.PI / 2 : headingRad(r);
      ctx.save(); ctx.translate(x, y); ctx.rotate(-hd);
      ctx.fillStyle = col; ctx.globalAlpha = r.out ? 0.45 : 1;
      ctx.beginPath(); ctx.moveTo(rad * 1.55, 0); ctx.lineTo(-rad * 0.6, rad * 0.95); ctx.lineTo(-rad * 0.6, -rad * 0.95); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.beginPath(); ctx.arc(x, y, rad * 0.78, 0, Math.PI * 2); ctx.fillStyle = col; ctx.globalAlpha = r.out ? 0.45 : 1; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = colors.paper; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = theme === "dark" || (theme === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "#111" : "#fff";
      ctx.font = "800 " + Math.round(rad * 0.95) + "px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(letter(r.id), x, y + 0.5);
    });
    drawScale(ctx);
    if (f) updateTeam(f); else updateTeam(null);
  }
  function drawPlanMarks(ctx) {
    var h = M.house;
    if (h.door) {
      ctx.fillStyle = colors.accent; ctx.font = "700 11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("door", xf.x(h.door[0]) + Math.max(10, xf.s * 0.5), xf.y(h.door[1]) + 8);
    }
    drawPads(ctx, h, xf, M.robots);
  }
  function drawScale(ctx) {
    var m = xf.s > 40 ? 2 : xf.s > 18 ? 5 : 10, w = m * xf.s, x = 14, y = cssH - 14;
    ctx.strokeStyle = colors.ink; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(x, y - 5); ctx.lineTo(x, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y - 5); ctx.stroke();
    ctx.fillStyle = colors.ink; ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(m + " m", x + w / 2, y - 4);
  }

  /* ---------- playback loop ---------- */
  var lastTs = 0, rafOn = false;
  function tick() { if (!rafOn) { rafOn = true; lastTs = performance.now(); requestAnimationFrame(loop); } }
  function loop(ts) {
    if (!M) { rafOn = false; return; }
    var dt = Math.min(0.25, (ts - lastTs) / 1000); lastTs = ts;
    if (M.ready >= 0 && !M.failed) {
      var edge = M.ready * DT;
      // Resume at the same simulated time in the exact-scene brain-only fallback.
      if (M.resumeAt) { if (edge >= M.resumeAt || M.done) { seek(M.resumeAt); M.resumeAt = 0; } else edge = -1; }
      if (edge < 0) { requestAnimationFrame(loop); return; }
      if (M.playing && M.t < edge) seek(Math.min(edge, M.t + dt * M.speed));
      else if (M.lastIdx < 0) seek(0);
      var end = M.done && M.t >= edge - 1e-6;
      if (end && !M.scoreShown) { M.scoreShown = true; M.playing = false; renderScore(); }
      syncUi(edge);
    } else if (M.id || !$("waiting").hidden) {
      var el_ = (performance.now() - M.started) / 1000;
      $("wait-clock").textContent = Math.floor(el_) + " s";
      var st = M.status, ss = $("sim-state");
      if (M.failed) ss.textContent = "";
      else if (st && st.state === "running" && num(st.sim_t)) ss.innerHTML = "<b>Simulating on AWS,</b> " + Math.floor(st.sim_t) + " s of " + MISSION_S + " s ready. The first 10 s arrive as one piece.";
      else if (st && st.state === "starting") ss.textContent = "Waiting for AWS to start the cloud machine.";
      else ss.textContent = M.source === "recorded" || M.source === 'saved' ? "Loading the recorded run." : "Sending the mission to AWS.";
    }
    draw();
    requestAnimationFrame(loop);
  }
  function syncUi(edge) {
    var sc = $("scrubber");
    if (document.activeElement !== sc || !scrubbing) sc.value = M.t;
    $("buffered").style.width = (100 * (edge / MISSION_S)) + "%";
    $("clock").textContent = fmt(M.t) + " / " + fmt(MISSION_S);
    syncPlay();
    var simT = M.status && num(M.status.sim_t) ? M.status.sim_t : edge;
    var s = $("sim-state");
    if (M.done) s.textContent = M.source === "recorded" ? "Recorded run, all " + MISSION_S + " s loaded." : "Simulation finished on AWS. All " + MISSION_S + " s are here.";
    else if (M.playing && M.t >= edge - 0.01) s.innerHTML = "<b>Waiting for the simulator.</b> " + Math.floor(Math.max(simT, edge)) + " s of " + MISSION_S + " s ready; the replay catches up as each 10 s arrives.";
    else s.textContent = (M.source === 'recorded' ? 'Loading recorded replay, ' : 'Simulating on AWS, ') + Math.floor(Math.max(simT, edge)) + " s of " + MISSION_S + " s ready.";
  }
  function syncPlay() {
    var b = $("play"), paused = !M || !M.playing;
    b.classList.toggle("paused", paused); b.setAttribute("aria-label", paused ? "Play" : "Pause");
  }
  function setSpeedButtons(x) { Array.prototype.forEach.call(document.querySelectorAll(".speed button"), function (b) { b.setAttribute("aria-pressed", String(+b.dataset.x === x)); }); }
  $("play").addEventListener("click", togglePlay);
  function togglePlay() {
    if (!M || M.ready < 0) return;
    if (!M.playing && M.done && M.t >= M.ready * DT - 1e-6) { seek(0); $("score").hidden = true; M.scoreShown = false; }
    M.playing = !M.playing; syncPlay();
  }
  $("stage").addEventListener("keydown", function (e) {
    if (e.key === " " || e.key === "k") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight" && M) { seek(M.t + 5); }
    if (e.key === "ArrowLeft" && M) { seek(M.t - 5); }
  });
  var scrubbing = false;
  $("scrubber").addEventListener("input", function () { if (!M) return; scrubbing = true; seek(+this.value); if (M.t < +this.value - 0.5) this.value = M.t; });
  $("scrubber").addEventListener("change", function () { scrubbing = false; if (M && M.t < M.ready * DT) { M.scoreShown = false; $("score").hidden = true; } });
  document.querySelector(".speed").addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b || !M) return; M.speed = +b.dataset.x; setSpeedButtons(M.speed);
  });

  /* ---------- score card ---------- */
  window.RescueLayout = {
    selection: function () { return { house: houseById(sel.house), robots: sel.robots, difficulty: sel.difficulty, layout: roomLayout }; },
    validate: function (layout) { return fetchJSON(API + '/validate', 30000, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({house:sel.house,robots:sel.robots,difficulty:sel.difficulty,layout:layout})}); },
    accept: function (reply) { if (!reply.house || reply.house.id !== sel.house) return; editedHouse = reply.house; roomLayout = reply.layout; scene = reply.placement; layoutValid = true; invalidateScene(); updateStartAvailability(); drawPreview(); $('scene-summary').textContent = scene.people.length + ' people on clear floor in your edited layout. Prepare its exact fallback before starting.'; },
    invalid: function () { layoutValid = false; updateStartAvailability(); },
    valid: function () { layoutValid = true; updateStartAvailability(); },
    prepare: function () { openEditor(); $('prepare-scene').click(); },
    reset: function () { resetScene(); drawPreview(); },
    mission: function () { return M; }
  };
  function renderScore() {
    var st = M.status || {}, s = st.score;
    var last = M.frames[M.ready];
    if (!s) {
      var firstOut = null;
      for (var i = 0; i <= M.ready; i++) if (M.frames[i] && M.frames[i].robots.length && M.frames[i].robots.every(function (r) { return r.out; })) { firstOut = M.frames[i].t; break; }
      s = { people_found: M.found.length, people_total: M.house.people, time_s: firstOut != null ? firstOut : MISSION_S,
        coverage_pct: 100 * M.unionCount / (M.house._floorCells || (M.house.floor_m2 || 1) / (CELL * CELL)),
        robots_out: last ? last.robots.filter(function (r) { return r.out; }).length : 0, false_confirmations: 0, derived: true };
    }
    $("s-people").innerHTML = s.people_found + " <small>of " + s.people_total + "</small>";
    $("s-time").innerHTML = (s.all_out != null ? s.all_out : (s.robots_out === M.robots && s.time_s < MISSION_S)) ? fmt(s.time_s) + " <small>all out</small>" : fmt(s.time_s) + " <small>time ran out</small>";
    $("s-cov").innerHTML = Math.round(s.coverage_pct) + " <small>%</small>";
    $("s-out").innerHTML = s.robots_out + " <small>of " + M.robots + "</small>";
    var missed = s.people_total - s.people_found;
    var fc = $("s-false");
    var notes = [];
    if (missed > 0) notes.push(missed + (missed === 1 ? ' person was not found.' : ' people were not found.'));
    if (s.false_confirmations) notes.push(s.false_confirmations + ' confirmation' + (s.false_confirmations > 1 ? 's did' : ' did') + ' not match a distinct person.');
    if (num(s.first_person_s)) notes.push("First person found at " + fmt(s.first_person_s) + ".");
    fc.hidden = !notes.length; fc.textContent = notes.join(" ");
    var cmp = $("s-compare");
    if (M.compare && M.compare.score) {
      var c = M.compare.score;
      cmp.textContent = "Same house with " + M.compare.robots + (M.compare.robots === 1 ? " robot" : " robots") + ": " + c.people_found + " of " + c.people_total + " people, " + Math.round(c.coverage_pct) + " % of the floor searched, " + ((c.all_out != null ? c.all_out : (c.robots_out === M.compare.robots && c.time_s < MISSION_S)) ? "all out at " + fmt(c.time_s) : "time ran out") + ".";
      cmp.hidden = false;
    } else cmp.hidden = true;
    $("score").hidden = false;
    var r = $("score").getBoundingClientRect();
    if (r.top > window.innerHeight - 80) $("score").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
    announce("Mission over. " + s.people_found + " of " + s.people_total + " people found.");
  }

  /* ---------- go ---------- */
  function redrawAll() { if (M && !$("mission").hidden) { if (M.union) { var t = M.t; M.lastIdx = -1; M.union = null; seek(t); } draw(); } else { drawThumbs(); drawPreview(); } }
  readColors();
  applyTheme();
  loadGallery();
})();

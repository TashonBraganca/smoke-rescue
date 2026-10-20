(function () {
  var $ = function (id) { return document.getElementById(id); };
  fetch('/api/health').then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
    .then(function (a) {
      $('status').textContent = a[0] ? 'healthy' : 'unhealthy';
      $('backend').textContent = a[1].backend || 'unknown';
      $('model').textContent = a[1].model || 'unknown';
    })
    .catch(function () { $('status').textContent = 'unreachable'; });

  $('try').addEventListener('submit', function (e) {
    e.preventDefault();
    var body = new FormData();
    body.append('frame', $('file').files[0]);
    $('out').textContent = 'sending';
    fetch('/api/vision/' + encodeURIComponent($('tool').value), { method: 'POST', body: body })
      .then(function (r) { return r.text().then(function (t) { return r.status + '\n' + t; }); })
      .then(function (t) { $('out').textContent = t; })
      .catch(function (err) { $('out').textContent = String(err); });
  });
})();

/* ============================================================
   MangaLens — API Utility (fetch wrapper)
   ============================================================ */

/* ── Frontend Logger ─────────────────────────────────────────
   Logs to console and stores recent entries for diagnostics.
   Levels: debug < info < warn < error
   ──────────────────────────────────────────────────────────── */
const Logger = (() => {
  const MAX_ENTRIES = 200;
  const _entries = [];
  const _level = { debug: 0, info: 1, warn: 2, error: 3 };
  const _minLevel = _level.info; // show info+ in console

  function _log(level, module, msg, ...data) {
    const ts = new Date().toISOString();
    const entry = { ts, level, module, msg, data };
    _entries.push(entry);
    if (_entries.length > MAX_ENTRIES) _entries.shift();

    if (_level[level] >= _minLevel) {
      const prefix = `[${ts.slice(11, 23)}] [${level.toUpperCase()}] [${module}]`;
      const fn = level === 'error' ? console.error
               : level === 'warn' ? console.warn
               : console.log;
      fn(prefix, msg, ...data);
    }
  }

  return {
    debug: (mod, msg, ...d) => _log('debug', mod, msg, ...d),
    info:  (mod, msg, ...d) => _log('info',  mod, msg, ...d),
    warn:  (mod, msg, ...d) => _log('warn',  mod, msg, ...d),
    error: (mod, msg, ...d) => _log('error', mod, msg, ...d),
    /** Return all stored log entries (for diagnostics). */
    dump: () => [..._entries],
  };
})();

const API = (() => {
  // FastAPI 백엔드가 프론트엔드 정적 파일도 함께 서빙 (동일 포트)
  const BASE = `${location.protocol}//${location.host}/api`;

  /**
   * Generic fetch wrapper with JSON parsing and error handling.
   * @param {string} path - e.g. "/health"
   * @param {RequestInit} opts
   * @returns {Promise<any>}
   */
  async function request(path, opts = {}) {
    const url = `${BASE}${path}`;
    const method = opts.method || 'GET';
    const headers = { ...(opts.headers || {}) };

    Logger.info('API', `${method} ${url}`);
    const t0 = performance.now();

    let res;
    try {
      res = await fetch(url, {
        ...opts,
        headers,
        credentials: 'same-origin',
      });
    } catch (networkErr) {
      Logger.error('API', `${method} ${url} — network error:`, networkErr.message);
      throw networkErr;
    }

    const elapsed = Math.round(performance.now() - t0);

    if (!res.ok) {
      let detail = `서버 오류 (${res.status})`;
      try {
        const body = await res.json();
        if (body.detail) detail = body.detail;
      } catch { /* ignore parse errors */ }
      Logger.error('API', `${method} ${url} — ${res.status} in ${elapsed}ms: ${detail}`);
      throw new Error(detail);
    }

    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const json = await res.json();
      Logger.info('API', `${method} ${url} — 200 in ${elapsed}ms`, json);
      return json;
    }
    Logger.info('API', `${method} ${url} — ${res.status} in ${elapsed}ms (non-JSON)`);
    return res;
  }

  /** GET JSON */
  function get(path) {
    return request(path);
  }

  /** POST JSON body */
  function postJSON(path, body) {
    return request(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /** POST FormData (no content-type — browser sets boundary) */
  function postForm(path, formData) {
    return request(path, {
      method: 'POST',
      body: formData,
    });
  }

  /** Build a download URL */
  function downloadUrl(path) {
    return `${BASE}${path}`;
  }

  /**
   * Open a WebSocket to /ws/progress/{taskId}.
   * Returns the WebSocket instance.
   */
  function wsProgress(taskId) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return new WebSocket(`${proto}://${location.host}/ws/progress/${encodeURIComponent(taskId)}`);
  }

  return { get, postJSON, postForm, downloadUrl, wsProgress };
})();

/* ============================================================
   Toast notifications
   ============================================================ */
const Toast = (() => {
  const container = () => document.getElementById('toast-container');

  function show(message, type = 'error') {
    const el = document.createElement('div');
    el.className = `toast toast--${type}`;
    el.textContent = message;
    container().appendChild(el);
    setTimeout(() => { el.remove(); }, 3000);
  }

  return {
    error: (msg) => show(msg, 'error'),
    success: (msg) => show(msg, 'success'),
    warn: (msg) => show(msg, 'warn'),
  };
})();

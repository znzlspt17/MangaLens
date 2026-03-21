/* ============================================================
   MangaLens — API Utility (fetch wrapper)
   ============================================================ */

const API = (() => {
  const BASE = '/api';

  /**
   * Generic fetch wrapper with JSON parsing and error handling.
   * @param {string} path - e.g. "/health"
   * @param {RequestInit} opts
   * @returns {Promise<any>}
   */
  async function request(path, opts = {}) {
    const url = `${BASE}${path}`;
    const headers = { ...(opts.headers || {}) };

    // Attach session cookie automatically (browser handles cookie header)
    // Attach API key headers if stored
    const deeplKey = sessionStorage.getItem('mangalens_deepl_key');
    const googleKey = sessionStorage.getItem('mangalens_google_key');
    if (deeplKey) headers['X-DeepL-Key'] = deeplKey;
    if (googleKey) headers['X-Google-Key'] = googleKey;

    const res = await fetch(url, {
      ...opts,
      headers,
      credentials: 'same-origin',
    });

    if (!res.ok) {
      let detail = `서버 오류 (${res.status})`;
      try {
        const body = await res.json();
        if (body.detail) detail = body.detail;
      } catch { /* ignore parse errors */ }
      throw new Error(detail);
    }

    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      return res.json();
    }
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

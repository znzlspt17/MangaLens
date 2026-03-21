/* ============================================================
   MangaLens — Progress Module (WebSocket)
   ============================================================ */

const Progress = (() => {
  let ws = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT = 3;
  let currentTaskId = null;

  /* --- DOM refs -------------------------------------------- */
  function el(id) { return document.getElementById(id); }

  /* --- Start tracking -------------------------------------- */
  function start(taskId, totalHint) {
    currentTaskId = taskId;
    reconnectAttempts = 0;

    // Reset UI
    updateUI({ progress: 0, total_images: totalHint || 0, completed_images: 0, failed_images: 0, status: 'queued' });

    connect(taskId);
  }

  /* --- WebSocket connection -------------------------------- */
  function connect(taskId) {
    if (ws) { ws.onclose = null; ws.close(); }

    ws = API.wsProgress(taskId);

    ws.addEventListener('open', () => {
      /* reconnectAttempts는 start()에서만 리셋 — 재연결 중 리셋하지 않음 */
    });

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        updateUI(data);

        if (['completed', 'partial', 'failed'].includes(data.status)) {
          stop();
          Result.show(taskId, data);
          App.showSection('result');
        }
      } catch { /* ignore malformed messages */ }
    });

    ws.addEventListener('close', () => {
      if (!currentTaskId || currentTaskId !== taskId) return;
      if (reconnectAttempts < MAX_RECONNECT) {
        reconnectAttempts++;
        const delay = Math.min(1000 * 2 ** reconnectAttempts, 8000);
        Toast.warn(`연결 끊김 — 재연결 시도 ${reconnectAttempts}/${MAX_RECONNECT}`);
        setTimeout(() => connect(taskId), delay);
      }
    });

    ws.addEventListener('error', () => {
      // error fires before close; close handler will deal with reconnect
    });
  }

  /* --- Update progress UI ---------------------------------- */
  function updateUI(data) {
    const pct = Math.round(data.progress ?? 0);
    el('progress-bar').style.width = `${pct}%`;
    el('progress-text').textContent = `${pct}%`;

    const wrapper = el('progress-bar').closest('.progress-bar-wrapper');
    if (wrapper) {
      wrapper.setAttribute('aria-valuenow', pct);
    }

    el('progress-completed').textContent = data.completed_images ?? 0;
    el('progress-total').textContent = data.total_images ?? 0;
    el('progress-failed').textContent = data.failed_images ?? 0;

    const statusMap = {
      queued: '대기 중…',
      processing: '처리 중…',
      completed: '완료',
      partial: '부분 완료',
      failed: '실패',
    };
    el('progress-status').textContent = statusMap[data.status] || data.status;
  }

  /* --- Stop / cleanup -------------------------------------- */
  function stop() {
    currentTaskId = null;
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }

  return { start, stop };
})();

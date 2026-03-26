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
    Logger.info('Progress', `Start tracking task=${taskId}, totalImages=${totalHint}`);
    currentTaskId = taskId;
    reconnectAttempts = 0;

    // Reset UI
    updateUI({ progress: 0, total_images: totalHint || 0, completed_images: 0, failed_images: 0, status: 'queued' });

    connect(taskId);
  }

  /* --- WebSocket connection -------------------------------- */
  function connect(taskId) {
    if (ws) { ws.onclose = null; ws.close(); }

    Logger.info('Progress', `WebSocket connecting for task=${taskId}`);
    ws = API.wsProgress(taskId);

    ws.addEventListener('open', () => {
      Logger.info('Progress', `WebSocket connected for task=${taskId}`);
    });

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        Logger.debug('Progress', `WS message: status=${data.status} progress=${data.progress}%`, data);
        updateUI(data);

        if (['completed', 'partial', 'failed'].includes(data.status)) {
          Logger.info('Progress', `Task ${taskId} terminal: status=${data.status}, completed=${data.completed_images}, failed=${data.failed_images}`);
          stop();
          Result.show(taskId, data);
          App.showSection('result');
        }
      } catch (err) {
        Logger.warn('Progress', 'Malformed WebSocket message:', event.data, err);
      }
    });

    ws.addEventListener('close', (event) => {
      Logger.warn('Progress', `WebSocket closed: code=${event.code} reason=${event.reason} task=${taskId}`);
      if (!currentTaskId || currentTaskId !== taskId) return;
      if (reconnectAttempts < MAX_RECONNECT) {
        reconnectAttempts++;
        const delay = Math.min(1000 * 2 ** reconnectAttempts, 8000);
        Logger.info('Progress', `Reconnecting ${reconnectAttempts}/${MAX_RECONNECT} in ${delay}ms`);
        Toast.warn(`연결 끊김 — 재연결 시도 ${reconnectAttempts}/${MAX_RECONNECT}`);
        setTimeout(() => connect(taskId), delay);
      } else {
        Logger.error('Progress', `Max reconnect attempts reached for task=${taskId}`);
        Toast.error('서버 연결이 끊어졌습니다. 페이지를 새로고침해 주세요.');
        el('progress-status').textContent = '연결 끊김';
      }
    });

    ws.addEventListener('error', (event) => {
      Logger.error('Progress', `WebSocket error for task=${taskId}`, event);
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
    Logger.debug('Progress', `Stopping tracking for task=${currentTaskId}`);
    currentTaskId = null;
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }

  return { start, stop };
})();

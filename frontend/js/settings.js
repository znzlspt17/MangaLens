/* ============================================================
   MangaLens — Settings Module
   ============================================================ */

const Settings = (() => {
  function el(id) { return document.getElementById(id); }

  /* --- Init ------------------------------------------------ */
  function init() {
    // Open / close modal
    document.getElementById('settings-btn').addEventListener('click', openModal);
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('modal-cancel').addEventListener('click', closeModal);

    // Banner "설정 열기" 버튼
    document.getElementById('banner-settings-btn')?.addEventListener('click', openModal);

    // Close on overlay click
    document.getElementById('settings-modal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeModal();
    });

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });

    // Save
    document.getElementById('settings-save').addEventListener('click', saveSettings);

    // Visibility toggles
    document.querySelectorAll('.toggle-vis').forEach((btn) => {
      btn.addEventListener('click', () => {
        const input = document.getElementById(btn.dataset.target);
        if (input.type === 'password') {
          input.type = 'text';
          btn.textContent = '🔒';
        } else {
          input.type = 'password';
          btn.textContent = '👁';
        }
      });
    });

    // 초기 API 키 배너 확인
    checkAndShowBanner();
  }

  /* --- Open modal ------------------------------------------ */
  async function openModal() {
    document.getElementById('settings-modal').classList.add('active');

    // Load current settings
    try {
      const data = await API.get('/settings');
      if (data.session_id) API.setSessionId(data.session_id);
      el('deepl-status').textContent = data.deepl_api_key ? `저장됨: ${data.deepl_api_key}` : '설정되지 않음';
      el('google-status').textContent = data.google_api_key ? `저장됨: ${data.google_api_key}` : '설정되지 않음';
    } catch {
      el('deepl-status').textContent = '불러오기 실패';
      el('google-status').textContent = '불러오기 실패';
    }

    // Load GPU info
    loadGPUInfo();
  }

  /* --- Close modal ----------------------------------------- */
  function closeModal() {
    document.getElementById('settings-modal').classList.remove('active');
    // 입력값 유지 — 취소 시 미완성 입력 유실 방지
  }

  /* --- Save settings --------------------------------------- */
  async function saveSettings() {
    const deeplKey = el('deepl-key').value.trim() || null;
    const googleKey = el('google-key').value.trim() || null;

    // Only send if at least one key is provided
    if (!deeplKey && !googleKey) {
      Toast.warn('API 키를 입력해주세요');
      return;
    }

    try {
      const body = {};
      if (deeplKey) body.deepl_api_key = deeplKey;
      if (googleKey) body.google_api_key = googleKey;
      const data = await API.postJSON('/settings', body);

      // session_id를 localStorage에 저장하여 이후 요청에 X-Session-Id 헤더로 활용
      if (data.session_id) API.setSessionId(data.session_id);

      el('deepl-status').textContent = data.deepl_api_key ? `저장됨: ${data.deepl_api_key}` : '설정되지 않음';
      el('google-status').textContent = data.google_api_key ? `저장됨: ${data.google_api_key}` : '설정되지 않음';

      Toast.success('API 키가 저장되었습니다');
      _setBanner(false);
      closeModal();
    } catch (err) {
      Toast.error(err.message);
    }
  }

  /* --- Banner --------------------------------------------- */
  function _setBanner(show) {
    const banner = document.getElementById('api-key-banner');
    if (banner) banner.style.display = show ? '' : 'none';
  }

  async function checkAndShowBanner() {
    try {
      const data = await API.get('/settings');
      if (data.session_id) API.setSessionId(data.session_id);
      _setBanner(!(data.deepl_api_key || data.google_api_key));
    } catch {
      _setBanner(true);
    }
  }

  /* --- GPU info -------------------------------------------- */
  async function loadGPUInfo() {
    try {
      const gpu = await API.get('/system/gpu');
      el('gpu-status-text').textContent = '연결됨';
      el('gpu-name').textContent = gpu.gpu_name || '-';
      el('gpu-backend').textContent = gpu.backend || '-';
      el('gpu-vram').textContent = gpu.vram_mb ? `${gpu.vram_mb} MB` : '-';
    } catch {
      el('gpu-status-text').textContent = '확인 실패';
    }
  }

  return { init, checkAndShowBanner };
})();

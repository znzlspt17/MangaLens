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

    // Close on overlay click
    document.getElementById('settings-modal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeModal();
    });

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });
  }

  /* --- Open modal ------------------------------------------ */
  function openModal() {
    document.getElementById('settings-modal').classList.add('active');
    loadGPUInfo();
  }

  /* --- Close modal ----------------------------------------- */
  function closeModal() {
    document.getElementById('settings-modal').classList.remove('active');
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

  return { init };
})();


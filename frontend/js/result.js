/* ============================================================
   MangaLens — Result Module
   ============================================================ */

const Result = (() => {
  let lastTaskId = null;
  let sliderDragging = false;
  let originalBlobUrl = null;

  function el(id) { return document.getElementById(id); }

  /* --- Show result ----------------------------------------- */
  function show(taskId, data) {
    lastTaskId = taskId;

    const completed = data.completed_images ?? 0;
    const failed = data.failed_images ?? 0;
    const total = data.total_images ?? 0;
    const status = data.status; // completed | partial | failed

    // Icon
    const iconEl = el('result-icon');
    if (status === 'completed') {
      iconEl.textContent = '✅';
      iconEl.className = 'result-icon result-icon--success';
    } else if (status === 'partial') {
      iconEl.textContent = '⚠️';
      iconEl.className = 'result-icon result-icon--partial';
    } else {
      iconEl.textContent = '❌';
      iconEl.className = 'result-icon result-icon--failed';
    }

    // Title
    const titles = { completed: '번역 완료', partial: '부분 완료', failed: '번역 실패' };
    el('result-title').textContent = titles[status] || '완료';

    // Summary
    const parts = [];
    if (total > 0) parts.push(`전체 ${total}장`);
    if (completed > 0) parts.push(`성공 ${completed}장`);
    if (failed > 0) parts.push(`실패 ${failed}장`);
    el('result-summary').textContent = parts.join(' · ');

    // Download button visibility
    el('download-btn').style.display = (status === 'failed') ? 'none' : '';

    // Comparison viewer
    const viewer = el('comparison-viewer');
    const unavailable = el('comparison-unavailable');
    const isSingle = total === 1 && status !== 'failed';

    if (isSingle) {
      const originalFile = Upload.getOriginalFile();
      if (originalFile) {
        if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
        originalBlobUrl = URL.createObjectURL(originalFile);
        const translatedUrl = API.downloadUrl(`/result/${encodeURIComponent(taskId)}`);
        showComparison(originalBlobUrl, translatedUrl);
        viewer.style.display = '';
      } else {
        viewer.style.display = 'none';
      }
      unavailable.style.display = 'none';
    } else if (total > 1) {
      viewer.style.display = 'none';
      unavailable.style.display = '';
    } else {
      viewer.style.display = 'none';
      unavailable.style.display = 'none';
    }
  }

  /* --- Comparison viewer ----------------------------------- */
  function showComparison(origUrl, transUrl) {
    const origImg = el('comparison-original-img');
    const transImg = el('comparison-translated-img');
    const overlay = el('comparison-overlay');
    const slider = el('comparison-slider');

    origImg.src = origUrl;
    transImg.src = transUrl;

    // Reset slider to center
    overlay.style.clipPath = 'inset(0 0 0 50%)';
    slider.style.left = '50%';
    slider.setAttribute('aria-valuenow', '50');
  }

  function initSliderEvents() {
    const slider = el('comparison-slider');
    const overlay = el('comparison-overlay');
    const container = el('comparison-container');

    function updatePosition(clientX) {
      const rect = container.getBoundingClientRect();
      let x = clientX - rect.left;
      x = Math.max(0, Math.min(x, rect.width));
      const pct = (x / rect.width) * 100;
      overlay.style.clipPath = `inset(0 0 0 ${pct}%)`;
      slider.style.left = `${pct}%`;
      slider.setAttribute('aria-valuenow', Math.round(pct).toString());
    }

    // Mouse events
    slider.addEventListener('mousedown', (e) => {
      e.preventDefault();
      sliderDragging = true;
    });

    container.addEventListener('mousemove', (e) => {
      if (!sliderDragging) return;
      updatePosition(e.clientX);
    });

    document.addEventListener('mouseup', () => {
      sliderDragging = false;
    });

    // Click to jump
    container.addEventListener('click', (e) => {
      updatePosition(e.clientX);
    });

    // Touch events
    slider.addEventListener('touchstart', (e) => {
      e.preventDefault();
      sliderDragging = true;
    });

    container.addEventListener('touchmove', (e) => {
      if (!sliderDragging) return;
      updatePosition(e.touches[0].clientX);
    });

    document.addEventListener('touchend', () => {
      sliderDragging = false;
    });

    // Keyboard support
    slider.addEventListener('keydown', (e) => {
      const step = 2;
      const rect = container.getBoundingClientRect();
      const currentLeft = parseFloat(slider.style.left) || 50;
      let newPct = currentLeft;

      if (e.key === 'ArrowLeft') {
        newPct = Math.max(0, currentLeft - step);
      } else if (e.key === 'ArrowRight') {
        newPct = Math.min(100, currentLeft + step);
      } else {
        return;
      }
      e.preventDefault();
      overlay.style.clipPath = `inset(0 0 0 ${newPct}%)`;
      slider.style.left = `${newPct}%`;
      slider.setAttribute('aria-valuenow', Math.round(newPct).toString());
    });

    // Make slider focusable for keyboard
    slider.setAttribute('tabindex', '0');
  }

  /* --- Init ------------------------------------------------ */
  function init() {
    el('download-btn').addEventListener('click', () => {
      if (!lastTaskId) return;
      const a = document.createElement('a');
      a.href = API.downloadUrl(`/result/${encodeURIComponent(lastTaskId)}`);
      a.download = '';
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    el('new-translate-btn').addEventListener('click', () => {
      Progress.stop();
      Upload.reset();
      App.showSection('upload');
    });

    initSliderEvents();
  }

  return { init, show };
})();

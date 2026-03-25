/* ============================================================
   MangaLens — Upload Module
   ============================================================ */

const Upload = (() => {
  const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif'];
  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

  let selectedFiles = []; // Array of File objects
  let lastOriginalFile = null;
  let _serverReady = false;

  /* --- DOM refs (lazy, cached after first call) ------------- */
  let _els = null;
  function els() {
    if (!_els) {
      _els = {
        dropzone: document.getElementById('dropzone'),
        fileInput: document.getElementById('file-input'),
        selectBtn: document.getElementById('file-select-btn'),
        grid: document.getElementById('file-preview-grid'),
        uploadBtn: document.getElementById('upload-btn'),
      };
    }
    return _els;
  }

  /* --- Init ------------------------------------------------- */
  function init() {
    const { dropzone, fileInput, selectBtn, uploadBtn } = els();

    // Click dropzone or select-btn → open file picker
    dropzone.addEventListener('click', (e) => {
      if (e.target === selectBtn || selectBtn.contains(e.target)) return;
      fileInput.click();
    });
    selectBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    // Keyboard accessibility
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    // File input change
    fileInput.addEventListener('change', () => {
      addFiles(fileInput.files);
      fileInput.value = '';
    });

    // Drag & Drop
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
    dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('drag-over'); });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      addFiles(e.dataTransfer.files);
    });

    // Upload button
    uploadBtn.addEventListener('click', startUpload);
  }

  /* --- Validation ------------------------------------------ */
  function validateFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      Toast.error(`지원하지 않는 파일 형식입니다: ${file.name}`);
      return false;
    }
    if (file.size > MAX_FILE_SIZE) {
      Toast.error(`파일 크기가 50MB를 초과합니다: ${file.name}`);
      return false;
    }
    return true;
  }

  /* --- Add files ------------------------------------------- */
  function addFiles(fileList) {
    for (const file of fileList) {
      if (validateFile(file)) {
        selectedFiles.push(file);
      }
    }
    renderPreviews();
    updateUploadBtn();
  }

  /* --- Remove file ----------------------------------------- */
  function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderPreviews();
    updateUploadBtn();
  }

  /* --- Render previews ------------------------------------- */
  function renderPreviews() {
    const { grid } = els();
    grid.innerHTML = '';
    selectedFiles.forEach((file, idx) => {
      const card = document.createElement('div');
      card.className = 'file-card';

      const img = document.createElement('img');
      img.className = 'file-card__img';
      img.alt = file.name;
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);

      const info = document.createElement('div');
      info.className = 'file-card__info';

      const name = document.createElement('div');
      name.className = 'file-card__name';
      name.textContent = file.name;
      name.title = file.name;

      const size = document.createElement('div');
      size.className = 'file-card__size';
      size.textContent = formatSize(file.size);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'file-card__remove';
      removeBtn.innerHTML = '✕';
      removeBtn.setAttribute('aria-label', `${file.name} 제거`);
      removeBtn.addEventListener('click', () => removeFile(idx));

      info.append(name, size);
      card.append(img, info, removeBtn);
      grid.appendChild(card);
    });
  }

  /* --- Upload ---------------------------------------------- */
  async function startUpload() {
    if (selectedFiles.length === 0) return;

    const { uploadBtn } = els();
    uploadBtn.disabled = true;

    // API 키 미설정 시 경고
    if (!Settings.hasApiKey()) {
      const ok = confirm(
        '⚠️ 번역 API 키가 설정되지 않았습니다.\n\n'
        + '키 없이 진행하면 일본어 원문이 그대로 출력됩니다.\n'
        + '계속하시겠습니까?\n\n'
        + '("취소" → 설정에서 API 키 입력)'
      );
      if (!ok) {
        uploadBtn.disabled = false;
        return;
      }
    }

    try {
      lastOriginalFile = selectedFiles.length === 1 ? selectedFiles[0] : null;
      let result;
      if (selectedFiles.length === 1) {
        const fd = new FormData();
        fd.append('file', selectedFiles[0]);
        result = await API.postForm('/upload', fd);
      } else {
        const fd = new FormData();
        selectedFiles.forEach((f) => fd.append('files', f));
        result = await API.postForm('/upload/bulk', fd);
      }
      // Start progress tracking
      Progress.start(result.task_id, selectedFiles.length);
      App.showSection('progress');
    } catch (err) {
      Toast.error(err.message);
      uploadBtn.disabled = false;
    }
  }

  /* --- Reset state ----------------------------------------- */
  function reset() {
    selectedFiles = [];
    renderPreviews();
    updateUploadBtn();
  }

  /* --- Helpers --------------------------------------------- */
  function updateUploadBtn() {
    const btn = els().uploadBtn;
    const count = selectedFiles.length;
    if (count === 0) {
      btn.textContent = '번역 시작';
      btn.disabled = true;
      btn.title = '';
    } else if (!_serverReady) {
      btn.textContent = count > 1 ? `번역 시작 (${count}장)` : '번역 시작';
      btn.disabled = true;
      btn.title = '서버 준비 중… 잠시 후 다시 시도하세요';
    } else {
      btn.textContent = count > 1 ? `번역 시작 (${count}장)` : '번역 시작';
      btn.disabled = false;
      btn.title = '';
    }
  }

  function setServerReady(ready) {
    _serverReady = ready;
    updateUploadBtn();
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function getOriginalFile() { return lastOriginalFile; }

  return { init, reset, getOriginalFile, setServerReady };
})();

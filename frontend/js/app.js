/* ============================================================
   MangaLens — App Initialization
   ============================================================ */

const App = (() => {
  const sections = ['upload', 'progress', 'result'];

  /* --- Show a specific section ----------------------------- */
  function showSection(name) {
    sections.forEach((s) => {
      const el = document.getElementById(`${s}-section`);
      if (el) el.classList.toggle('active', s === name);
    });
  }

  /* --- Theme toggle ---------------------------------------- */
  function initTheme() {
    const toggle = document.getElementById('theme-toggle');
    const stored = localStorage.getItem('mangalens_theme');

    if (stored) {
      document.documentElement.setAttribute('data-theme', stored);
    }
    updateThemeIcon();

    toggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const isDark = current === 'dark' ||
        (!current && window.matchMedia('(prefers-color-scheme: dark)').matches);
      const next = isDark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('mangalens_theme', next);
      updateThemeIcon();
    });
  }

  function updateThemeIcon() {
    const toggle = document.getElementById('theme-toggle');
    const theme = document.documentElement.getAttribute('data-theme');
    const isDark = theme === 'dark' ||
      (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    toggle.textContent = isDark ? '☀️' : '🌙';
  }

  /* --- Server health polling ------------------------------- */
  async function checkHealth() {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    try {
      const data = await API.get('/health');
      if (data.ready) {
        dot.className = 'status-dot status-dot--ready';
        const gpuName = data.gpu_info?.gpu_name || '';
        label.textContent = gpuName ? `Ready · ${gpuName}` : 'Ready';
      } else {
        dot.className = 'status-dot status-dot--not-ready';
        label.textContent = '모델 로딩 중…';
      }
    } catch {
      dot.className = 'status-dot status-dot--not-ready';
      label.textContent = '서버 연결 실패';
    }
  }

  /* --- Cancel button --------------------------------------- */
  function initCancel() {
    document.getElementById('cancel-btn').addEventListener('click', () => {
      Progress.stop();
      Upload.reset();
      showSection('upload');
    });
  }

  /* --- Boot ------------------------------------------------ */
  function boot() {
    initTheme();
    Upload.init();
    Result.init();
    Settings.init();
    initCancel();

    // Initial health check + periodic
    checkHealth();
    setInterval(checkHealth, 30000);
  }

  document.addEventListener('DOMContentLoaded', boot);

  return { showSection };
})();

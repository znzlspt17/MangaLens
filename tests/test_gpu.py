"""GPU detection tests for MangaLens (server/gpu.py).

Tests cover:
- CUDA detection path (nvidia-smi mock)
- ROCm detection path (rocm-smi mock)
- CPU fallback when no GPU
- force_backend parameter
- ROCm gfx1201 env var auto-setting (HSA_OVERRIDE_GFX_VERSION)
- get_device() / get_vram_mb() helpers
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from server.gpu import (
    GPUInfo,
    _detect_cpu,
    _detect_nvidia,
    _detect_rocm,
    _parse_rocm_gfx_arch,
    detect_gpu,
    get_device,
    get_vram_mb,
)


# ---------------------------------------------------------------------------
# Helpers — reset cached GPU state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_gpu_cache():
    """Clear the module-level GPU cache before each test."""
    import server.gpu as gpu_mod
    gpu_mod._cached_info = None
    yield
    gpu_mod._cached_info = None


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

class TestCPUFallback:
    def test_detect_cpu_returns_cpu_info(self):
        info = _detect_cpu()
        assert info.backend == "cpu"
        assert info.device == "cpu"
        assert info.gpu_name == "CPU"
        assert info.vram_mb == 0
        assert info.driver_version == ""

    @patch("server.gpu._detect_nvidia", return_value=None)
    @patch("server.gpu._detect_rocm", return_value=None)
    def test_auto_falls_back_to_cpu(self, mock_rocm, mock_nvidia):
        info = detect_gpu(force_backend="auto")
        assert info.backend == "cpu"
        assert info.device == "cpu"

    @patch("server.gpu._detect_nvidia", return_value=None)
    @patch("server.gpu._detect_rocm", return_value=None)
    def test_force_cpu_skips_gpu_detection(self, mock_rocm, mock_nvidia):
        info = detect_gpu(force_backend="cpu")
        assert info.backend == "cpu"
        mock_nvidia.assert_not_called()
        mock_rocm.assert_not_called()


# ---------------------------------------------------------------------------
# CUDA (NVIDIA) path
# ---------------------------------------------------------------------------

class TestNvidiaDetection:
    @patch("server.gpu.shutil.which", return_value=None)
    def test_no_nvidia_smi_returns_none(self, mock_which):
        result = _detect_nvidia()
        assert result is None

    @patch("server.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("server.gpu._run_cmd", return_value="NVIDIA RTX 4090, 24564, 560.35")
    def test_nvidia_smi_found_but_no_torch_cuda(self, mock_cmd, mock_which):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _detect_nvidia()
            assert result is None

    @patch("server.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("server.gpu._run_cmd", return_value="NVIDIA RTX 4090, 24564, 560.35")
    def test_nvidia_smi_with_torch_cuda(self, mock_cmd, mock_which):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "13.0"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _detect_nvidia()
            assert result is not None
            assert result.backend == "cuda"
            assert result.device == "cuda"
            assert result.gpu_name == "NVIDIA RTX 4090"
            assert result.vram_mb == 24564
            assert "CUDA 13.0" in result.driver_version
            assert "560.35" in result.driver_version

    @patch("server.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("server.gpu._run_cmd", return_value=None)
    def test_nvidia_smi_cmd_fails(self, mock_cmd, mock_which):
        result = _detect_nvidia()
        assert result is None


# ---------------------------------------------------------------------------
# force_backend CUDA — falls back to CPU if no GPU
# ---------------------------------------------------------------------------

class TestForceBackend:
    @patch("server.gpu._detect_nvidia", return_value=None)
    def test_force_cuda_no_gpu_falls_back_cpu(self, mock_nv):
        info = detect_gpu(force_backend="cuda")
        assert info.backend == "cpu"

    @patch("server.gpu._detect_rocm", return_value=None)
    def test_force_rocm_no_gpu_falls_back_cpu(self, mock_rocm):
        info = detect_gpu(force_backend="rocm")
        assert info.backend == "cpu"

    def test_force_cuda_with_gpu(self):
        cuda_info = GPUInfo(
            backend="cuda", device="cuda",
            gpu_name="RTX 4090", vram_mb=24000,
            driver_version="CUDA 13.0",
        )
        with patch("server.gpu._detect_nvidia", return_value=cuda_info):
            info = detect_gpu(force_backend="cuda")
            assert info.backend == "cuda"
            assert info.gpu_name == "RTX 4090"


# ---------------------------------------------------------------------------
# ROCm path
# ---------------------------------------------------------------------------

class TestROCmDetection:
    @patch("server.gpu.shutil.which", return_value=None)
    def test_no_rocm_smi_returns_none(self, mock_which):
        result = _detect_rocm()
        assert result is None

    @patch("server.gpu.shutil.which", return_value="/usr/bin/rocm-smi")
    @patch("server.gpu._run_cmd")
    def test_rocm_smi_no_hip_torch(self, mock_cmd, mock_which):
        mock_cmd.return_value = "Card: AMD Radeon RX 9070 XT"
        mock_torch = MagicMock()
        mock_torch.version.hip = None  # Not ROCm build
        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("server.gpu._parse_rocm_gfx_arch", return_value=""):
            result = _detect_rocm()
            assert result is None

    @patch("server.gpu.shutil.which", return_value="/usr/bin/rocm-smi")
    @patch("server.gpu._run_cmd")
    @patch("server.gpu._parse_rocm_gfx_arch", return_value="1201")
    def test_rocm_smi_with_hip_torch(self, mock_gfx, mock_cmd, mock_which):
        def cmd_side_effect(cmd, *args, **kwargs):
            if "--showproductname" in cmd:
                return "Card series: AMD Radeon RX 9070 XT"
            if "--showmeminfo" in cmd:
                return "GPU[0] : VRAM Total: 16384 MB"
            return None

        mock_cmd.side_effect = cmd_side_effect
        mock_torch = MagicMock()
        mock_torch.version.hip = "6.3.0"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _detect_rocm()
            assert result is not None
            assert result.backend == "rocm"
            assert result.device == "cuda"  # ROCm uses cuda API
            assert "ROCm 6.3.0" in result.driver_version
            assert "gfx1201" in result.driver_version


# ---------------------------------------------------------------------------
# ROCm gfx1201 environment variable auto-set
# ---------------------------------------------------------------------------

class TestROCmGfx1201EnvVars:
    @patch("server.gpu.shutil.which", return_value="/usr/bin/rocm-smi")
    @patch("server.gpu._run_cmd")
    @patch("server.gpu._parse_rocm_gfx_arch", return_value="1201")
    def test_gfx1201_sets_hsa_override(self, mock_gfx, mock_cmd, mock_which):
        mock_cmd.return_value = None
        mock_torch = MagicMock()
        mock_torch.version.hip = "6.3.0"

        # Clear env vars if set
        env_backup = {}
        for key in ("HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            with patch.dict("sys.modules", {"torch": mock_torch}):
                _detect_rocm()
                assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "12.0.1"
                assert os.environ.get("PYTORCH_ROCM_ARCH") == "gfx1201"
        finally:
            # Restore env
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val


# ---------------------------------------------------------------------------
# _parse_rocm_gfx_arch
# ---------------------------------------------------------------------------

class TestParseGfxArch:
    @patch("server.gpu._run_cmd", return_value=None)
    def test_rocminfo_not_found(self, mock_cmd):
        assert _parse_rocm_gfx_arch() == ""

    @patch("server.gpu._run_cmd", return_value="  Name:                    gfx1201\n  Name: amdgcn-amd-amdhsa--gfx1201")
    def test_parse_gfx1201(self, mock_cmd):
        result = _parse_rocm_gfx_arch()
        assert result == "1201"

    @patch("server.gpu._run_cmd", return_value="Agent 1\n  Name: host\n  Something else")
    def test_no_gfx_in_output(self, mock_cmd):
        result = _parse_rocm_gfx_arch()
        assert result == ""


# ---------------------------------------------------------------------------
# get_device / get_vram_mb
# ---------------------------------------------------------------------------

class TestDeviceHelpers:
    @patch("server.gpu._detect_nvidia", return_value=None)
    @patch("server.gpu._detect_rocm", return_value=None)
    def test_get_device_cpu(self, mock_rocm, mock_nvidia):
        assert get_device() == "cpu"

    @patch("server.gpu._detect_nvidia", return_value=None)
    @patch("server.gpu._detect_rocm", return_value=None)
    def test_get_vram_mb_cpu(self, mock_rocm, mock_nvidia):
        assert get_vram_mb() == 0

    def test_get_device_cuda(self):
        cuda_info = GPUInfo(
            backend="cuda", device="cuda",
            gpu_name="RTX 4090", vram_mb=24000,
            driver_version="CUDA 13.0",
        )
        import server.gpu as gpu_mod
        gpu_mod._cached_info = cuda_info
        assert get_device() == "cuda"
        assert get_vram_mb() == 24000


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:
    def test_detect_gpu_caches_result(self):
        """Second call should return cached result without re-detecting."""
        import server.gpu as gpu_mod

        first_info = GPUInfo(
            backend="cpu", device="cpu",
            gpu_name="CPU", vram_mb=0, driver_version="",
        )
        gpu_mod._cached_info = first_info

        # Even force_backend differs, cached result is returned
        info = detect_gpu(force_backend="cuda")
        assert info is first_info

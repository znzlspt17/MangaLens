"""GPU auto-detection module for MangaLens.

Detects CUDA (NVIDIA) or ROCm (AMD) GPU at server startup,
caches the result, and provides device/VRAM helpers.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPUInfo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUInfo:
    """Immutable snapshot of detected GPU environment."""

    backend: str  # "cuda" | "rocm" | "cpu"
    device: str  # "cuda" | "cpu"
    gpu_name: str  # e.g. "NVIDIA RTX 4090", "AMD Radeon RX 9070 XT"
    vram_mb: int  # VRAM in megabytes (0 for CPU)
    driver_version: str  # CUDA version, ROCm version, or ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_nvidia() -> GPUInfo | None:
    """Try to detect NVIDIA GPU via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None

    # Query GPU name, VRAM (MiB), CUDA driver version
    out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None

    try:
        # Take first GPU line
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_name = parts[0]
        vram_mb = int(float(parts[1]))
        driver_ver = parts[2] if len(parts) > 2 else ""
    except (IndexError, ValueError):
        return None

    # Verify torch CUDA availability
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "nvidia-smi found but torch.cuda.is_available() is False"
            )
            return None
        cuda_version = torch.version.cuda or ""
    except ImportError:
        logger.warning("PyTorch not installed; cannot verify CUDA")
        return None

    return GPUInfo(
        backend="cuda",
        device="cuda",
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        driver_version=f"CUDA {cuda_version} (driver {driver_ver})",
    )


def _parse_rocminfo() -> tuple[str, str]:
    """Parse rocminfo to get (gfx_arch, marketing_name).

    Returns e.g. ("gfx1201", "AMD Radeon RX 9070 XT").
    Falls back to ("", "AMD GPU") on failure.
    """
    out = _run_cmd(["rocminfo"])
    if not out:
        return "", "AMD GPU"

    gfx_arch = ""
    marketing_name = "AMD GPU"

    # Find GPU agent section (has "Device Type:  GPU")
    in_gpu_agent = False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:") and "gfx" in stripped:
            gfx_arch = "gfx" + stripped.split("gfx")[-1].strip()
            in_gpu_agent = True
        elif in_gpu_agent and stripped.startswith("Marketing Name:"):
            name = stripped.split(":", 1)[-1].strip()
            if name:
                marketing_name = name
            break

    return gfx_arch, marketing_name


def _detect_rocm() -> GPUInfo | None:
    """Try to detect AMD ROCm GPU.

    Uses rocminfo as primary source (works in WSL2 where rocm-smi
    may fail due to missing amdgpu kernel module).
    VRAM is queried from PyTorch when rocm-smi is unavailable.
    """
    # Primary detection via rocminfo (works on both native and WSL2)
    if not shutil.which("rocminfo"):
        return None

    gfx_arch, gpu_name = _parse_rocminfo()
    if not gfx_arch:
        return None

    # Auto-set environment variables for gfx1201 (RDNA 4)
    if "1201" in gfx_arch:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
        logger.info(
            "ROCm %s detected — set HSA_OVERRIDE_GFX_VERSION=12.0.1, "
            "PYTORCH_ROCM_ARCH=gfx1201",
            gfx_arch,
        )

    # Try VRAM from rocm-smi first (native Linux)
    vram_mb = 0
    vram_out = _run_cmd(["rocm-smi", "--showmeminfo", "vram"])
    if vram_out:
        for line in vram_out.splitlines():
            if "Total" in line:
                try:
                    parts = line.split()
                    for p in parts:
                        if p.replace(".", "").isdigit():
                            val = int(float(p))
                            if val > 100_000:
                                vram_mb = val // (1024 * 1024)
                            else:
                                vram_mb = val
                            break
                except (ValueError, IndexError):
                    pass

    # Verify torch HIP availability
    try:
        import torch

        hip_available = getattr(torch.version, "hip", None) is not None
        if not hip_available:
            logger.warning(
                "rocminfo found GPU but torch is not built with ROCm/HIP support"
            )
            return None
        rocm_version = torch.version.hip or ""

        # Fall back to PyTorch for VRAM (WSL2: rocm-smi may not work)
        if vram_mb == 0 and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_mb = props.total_mem // (1024 * 1024)
    except ImportError:
        logger.warning("PyTorch not installed; cannot verify ROCm")
        return None

    return GPUInfo(
        backend="rocm",
        device="cuda",  # ROCm PyTorch uses cuda API
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        driver_version=f"ROCm {rocm_version} ({gfx_arch})",
    )


def _detect_cpu() -> GPUInfo:
    """CPU fallback."""
    return GPUInfo(
        backend="cpu",
        device="cpu",
        gpu_name="CPU",
        vram_mb=0,
        driver_version="",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_info: GPUInfo | None = None


def detect_gpu(force_backend: str = "auto") -> GPUInfo:
    """Detect GPU environment and cache the result.

    Args:
        force_backend: "auto" | "cuda" | "rocm" | "cpu".
            When "auto", detection order is CUDA → ROCm → CPU.
    """
    global _cached_info
    if _cached_info is not None:
        return _cached_info

    if force_backend == "cuda":
        info = _detect_nvidia()
        if info is None:
            logger.error("CUDA backend forced but no NVIDIA GPU detected")
            info = _detect_cpu()
    elif force_backend == "rocm":
        info = _detect_rocm()
        if info is None:
            logger.error("ROCm backend forced but no AMD GPU detected")
            info = _detect_cpu()
    elif force_backend == "cpu":
        info = _detect_cpu()
    else:
        # auto: try CUDA first, then ROCm, then CPU
        info = _detect_nvidia()
        if info is None:
            info = _detect_rocm()
        if info is None:
            info = _detect_cpu()

    _cached_info = info

    # Log detection result
    if info.backend == "cpu":
        logger.warning("No GPU detected — running on CPU (slow)")
    else:
        logger.info(
            "GPU detected: %s | %s | VRAM %d MB | %s",
            info.backend.upper(),
            info.gpu_name,
            info.vram_mb,
            info.driver_version,
        )

    return info


def get_device() -> str:
    """Return torch device string ('cuda' or 'cpu')."""
    info = detect_gpu()
    return info.device


def get_vram_mb() -> int:
    """Return VRAM in MB. Used for Semaphore limit decisions."""
    info = detect_gpu()
    return info.vram_mb


def get_gpu_info() -> GPUInfo:
    """Return the cached GPUInfo (runs detection if not yet called)."""
    return detect_gpu()


def reset_cache() -> None:
    """Reset cached GPU info (for testing purposes)."""
    global _cached_info
    _cached_info = None

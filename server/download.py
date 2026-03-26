"""Download models and fonts required by MangaLens.

Usage:
    python -m server.download
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx

from server.config import settings

TIMEOUT = 300  # seconds

MODELS: list[dict[str, str]] = [
    {
        "name": "comic-text-detector",
        "filename": "comictextdetector.pt",
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt",
        "dest_dir": "model",
    },
    {
        "name": "Real-ESRGAN x2plus",
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "dest_dir": "model",
    },
    {
        "name": "Real-ESRGAN x4plus",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "dest_dir": "model",
    },
    {
        "name": "Real-ESRGAN x4plus-anime6B",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "dest_dir": "model",
    },
    {
        "name": "LaMa big-lama",
        "filename": "big-lama.pt",
        "url": "https://huggingface.co/signature-ai/big-lama/resolve/main/big-lama.pt",
        "dest_dir": "model",
    },
]

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf"
FONT_FILENAME = "NotoSansKR-Regular.ttf"

GOOGLE_FONTS_ZIP_URL = "https://fonts.google.com/download?family=Noto+Sans+KR"


def _format_size(n_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with progress output."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        if total:
            sys.stdout.write(f"downloading {_format_size(total)}... ")
        else:
            sys.stdout.write("downloading... ")
        sys.stdout.flush()

        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65_536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(f"\r  downloading {_format_size(total)}... {pct}%")
                    sys.stdout.flush()

        if total:
            sys.stdout.write(f"\r  downloading {_format_size(total)}... ")
            sys.stdout.flush()


def _download_font(font_dir: Path) -> None:
    """Download Noto Sans KR font.

    Tries the direct GitHub URL first. Falls back to Google Fonts ZIP.
    """
    dest = font_dir / FONT_FILENAME
    if dest.exists():
        print("✓ already exists")
        return

    font_dir.mkdir(parents=True, exist_ok=True)

    # Try direct .ttf first
    try:
        _download_file(FONT_URL, dest)
        print("✓ done")
        return
    except httpx.HTTPError:
        pass

    # Fallback: Google Fonts ZIP
    sys.stdout.write("downloading (zip)... ")
    sys.stdout.flush()
    with httpx.stream("GET", GOOGLE_FONTS_ZIP_URL, follow_redirects=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        data = b""
        for chunk in resp.iter_bytes(chunk_size=65_536):
            data += chunk

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Find a regular .ttf
        candidates = [n for n in zf.namelist() if n.endswith(".ttf")]
        if not candidates:
            print("✗ no .ttf found in ZIP")
            return
        # Prefer Regular weight
        chosen = next((c for c in candidates if "Regular" in c), candidates[0])
        with open(dest, "wb") as f:
            f.write(zf.read(chosen))

    print("✓ done")


def download_all() -> None:
    """Download all required models and fonts."""
    model_dir = Path(settings.model_cache_dir)
    font_dir = Path(settings.font_dir)
    total = len(MODELS) + 2  # +1 font, +1 manga-ocr info

    for idx, item in enumerate(MODELS, start=1):
        dest = model_dir / item["filename"]
        sys.stdout.write(f"[{idx}/{total}] {item['name']} ({item['filename']})... ")
        sys.stdout.flush()
        if dest.exists():
            print("✓ already exists")
            continue
        try:
            _download_file(item["url"], dest)
            print("✓ done")
        except httpx.HTTPError as exc:
            print(f"✗ failed: {exc}")

    # Font
    idx = len(MODELS) + 1
    sys.stdout.write(f"[{idx}/{total}] Noto Sans KR font... ")
    sys.stdout.flush()
    _download_font(font_dir)

    # manga-ocr info
    idx = total
    print(f"[{idx}/{total}] manga-ocr (auto-download on first use)... skipped")

    print("\nAll models ready!")


if __name__ == "__main__":
    download_all()

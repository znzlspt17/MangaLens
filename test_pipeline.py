#!/usr/bin/env python3
"""MangaLens — 서버 없이 파이프라인을 직접 실행하는 테스트 스크립트.

사용법:
  uv run python test_pipeline.py                   # 1.png 단일 이미지
  uv run python test_pipeline.py --image 5         # 특정 번호 이미지
  uv run python test_pipeline.py --all             # 1~16.png 순차 처리
  uv run python test_pipeline.py --images 1 3 7    # 지정 번호 여러 장
  uv run python test_pipeline.py --out /tmp/out    # 출력 경로 지정
  uv run python test_pipeline.py --lang EN         # 번역 대상 언어 변경 (기본: KO)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# ─── 컬러 로그 포맷 ──────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[0;37m",
        logging.INFO:     "\033[0;36m",
        logging.WARNING:  "\033[1;33m",
        logging.ERROR:    "\033[0;31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


# ─── 단일 이미지 처리 ─────────────────────────────────────────────────────────

async def _process_one(
    image_path: Path,
    out_dir: Path,
    target_lang: str,
    source_lang: str,
) -> tuple[bool, float]:
    """이미지 한 장을 파이프라인으로 처리합니다.

    Returns:
        (성공 여부, 소요 시간(s))
    """
    from server.pipeline.orchestrator import run_pipeline, UserTranslationSettings

    log = logging.getLogger("test_pipeline")
    settings = UserTranslationSettings(target_lang=target_lang, source_lang=source_lang)

    log.info("처리 시작: %s", image_path.name)
    t0 = time.monotonic()
    try:
        result = await run_pipeline(
            image_path=image_path,
            settings=settings,
            output_dir=out_dir / image_path.stem,
        )
        elapsed = time.monotonic() - t0
        log.info(
            "\033[1;32m완료\033[0m  %s  →  %s  (%.1f 초)",
            image_path.name,
            result.translated_image_path,
            elapsed,
        )
        return True, elapsed
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception("\033[0;31m실패\033[0m  %s  (%.1f 초): %s", image_path.name, elapsed, exc)
        return False, elapsed


# ─── 메인 ────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="서버 없이 MangaLens 파이프라인을 직접 실행합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image",  type=int, metavar="N",
                       help="resources/test/N.png 단일 처리 (기본: 1)")
    group.add_argument("--images", type=int, nargs="+", metavar="N",
                       help="지정 번호 여러 장 처리")
    group.add_argument("--all",    action="store_true",
                       help="resources/test/1~16.png 전체 처리")
    parser.add_argument("--out",   default="output/test_pipeline",
                        help="결과 저장 디렉토리 (기본: output/test_pipeline)")
    parser.add_argument("--lang",  default="KO",
                        help="번역 대상 언어 코드 (기본: KO)")
    parser.add_argument("--src",   default="JA",
                        help="원본 언어 코드 (기본: JA)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG 로그 출력")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("test_pipeline")

    # 대상 이미지 목록 결정
    test_dir = Path("resources/test")
    if args.all:
        images = sorted(test_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    elif args.images:
        images = [test_dir / f"{n}.png" for n in args.images]
    else:
        n = args.image if args.image else 1
        images = [test_dir / f"{n}.png"]

    missing = [p for p in images if not p.exists()]
    if missing:
        log.error("이미지를 찾을 수 없습니다: %s", [str(p) for p in missing])
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU 정보 출력
    try:
        from server.gpu import detect_gpu
        gpu = detect_gpu()
        log.info("GPU: %s (%s) — VRAM %d MB", gpu.gpu_name, gpu.backend, gpu.vram_mb)
    except Exception:
        log.warning("GPU 감지 실패 — CPU 사용")

    print(f"\n\033[1m=== MangaLens 파이프라인 직접 실행 ({len(images)}장) ===\033[0m\n")

    passed = 0
    failed = 0
    total_time = 0.0

    for img_path in images:
        ok, elapsed = await _process_one(img_path, out_dir, args.lang, args.src)
        total_time += elapsed
        if ok:
            passed += 1
        else:
            failed += 1
        print()  # 이미지 간 구분선

    # 요약
    print("\033[1m=== 결과 요약 ===\033[0m")
    print(f"  총 이미지  : {len(images)}장")
    print(f"  \033[1;32m성공\033[0m       : {passed}장")
    if failed:
        print(f"  \033[0;31m실패\033[0m       : {failed}장")
    print(f"  총 소요    : {total_time:.1f}초  (평균 {total_time/len(images):.1f}초/장)")
    print(f"  결과 위치  : {out_dir.resolve()}/\n")

    # 결과 파일 목록
    result_files = sorted(out_dir.rglob("*_translated.*"))
    if result_files:
        print("  저장된 결과:")
        for f in result_files:
            size_kb = f.stat().st_size // 1024
            print(f"    {f.relative_to(out_dir)}  ({size_kb} KB)")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

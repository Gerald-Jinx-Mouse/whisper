#!/usr/bin/env python3
"""Batch-transcribe the Trading video library with Whisper.

Writes a .txt and .srt beside every video under --root. Resume-safe: a file
whose siblings already exist and whose .txt is non-empty is skipped, so the
run can be interrupted and restarted without losing work.

The model is loaded once and reused for the whole batch; per-file try/except
keeps one bad video from killing the run.

    uv run python transcribe_trading.py --video /path/to/one.mp4   # smoke test
    uv run python transcribe_trading.py                            # full batch
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import whisper
from whisper.utils import get_writer

ROOT = Path("/home/jerry/Projects/Trading")
LOG_PATH = Path(__file__).resolve().parent / "logs" / "transcribe_trading.log"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
OUTPUT_FORMATS = ("txt", "srt")

log = logging.getLogger("transcribe_trading")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def find_videos(root: Path) -> list[Path]:
    """Every video under root, sorted so --limit is reproducible."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def outputs_for(video: Path) -> list[Path]:
    return [video.with_suffix(f".{fmt}") for fmt in OUTPUT_FORMATS]


def already_done(video: Path) -> bool:
    """True when every output exists and the .txt has content.

    The size check means a truncated file left by a hard kill gets redone
    rather than skipped forever.
    """
    outs = outputs_for(video)
    return all(p.exists() for p in outs) and outs[0].stat().st_size > 0


def transcribe_one(model, video: Path) -> None:
    # verbose=None silences both the per-segment dump and the tqdm bar; the
    # batch's own per-file logging is the progress signal.
    result = model.transcribe(str(video), language="en", verbose=None)
    for fmt in OUTPUT_FORMATS:
        # output_dir is the video's own parent, so transcripts land beside it
        get_writer(fmt, str(video.parent))(result, str(video), {})


def media_duration(video: Path) -> float | None:
    """Source length in seconds, for the realtime factor. Best effort."""
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="directory to walk (default: %(default)s)")
    ap.add_argument("--video", type=Path, help="transcribe this single file instead of walking --root")
    ap.add_argument("--model", default="turbo", help="whisper model name (default: %(default)s)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="torch device (default: %(default)s)")
    ap.add_argument("--limit", type=int, help="stop after N transcriptions")
    ap.add_argument("--dry-run", action="store_true", help="list what would be done, load nothing")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.video:
        if not args.video.is_file():
            log.error("no such file: %s", args.video)
            return 2
        videos = [args.video]
    else:
        if not args.root.is_dir():
            log.error("no such directory: %s", args.root)
            return 2
        videos = find_videos(args.root)

    pending = [v for v in videos if not already_done(v)]
    skipped = len(videos) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]

    log.info("=" * 72)
    log.info("found %d video(s) · %d already done · %d to transcribe",
             len(videos), skipped, len(pending))
    log.info("model=%s device=%s formats=%s", args.model, args.device, ",".join(OUTPUT_FORMATS))

    if args.dry_run:
        for v in pending:
            log.info("WOULD TRANSCRIBE  %s", v)
        log.info("dry run: nothing written")
        return 0

    if not pending:
        log.info("nothing to do")
        return 0

    log.info("loading model %r on %s ...", args.model, args.device)
    t0 = time.monotonic()
    model = whisper.load_model(args.model, device=args.device)
    log.info("model ready in %.1fs", time.monotonic() - t0)

    done = failed = 0
    batch_start = time.monotonic()

    for i, video in enumerate(pending, 1):
        log.info("[%d/%d] START  %s", i, len(pending), video)
        started = time.monotonic()
        try:
            transcribe_one(model, video)
        except KeyboardInterrupt:
            log.warning("interrupted by user — rerun to resume")
            raise
        except Exception:
            failed += 1
            log.error("[%d/%d] FAILED %s\n%s", i, len(pending), video, traceback.format_exc())
            continue

        elapsed = time.monotonic() - started
        duration = media_duration(video)
        speed = f" · {duration / elapsed:.1f}x realtime" if duration and elapsed > 0 else ""
        size = outputs_for(video)[0].stat().st_size
        done += 1
        log.info("[%d/%d] OK     %s · %.1fs%s · %d bytes",
                 i, len(pending), video.name, elapsed, speed, size)

    total = time.monotonic() - batch_start
    log.info("-" * 72)
    log.info("found %d · skipped %d · transcribed %d · failed %d · %.1f min total",
             len(videos), skipped, done, failed, total / 60)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

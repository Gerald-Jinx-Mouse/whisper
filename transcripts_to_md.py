#!/usr/bin/env python3
"""Render a Markdown transcript beside every video that already has an .srt.

Reads the .srt written by transcribe_trading.py and groups its segments into
readable paragraphs anchored with [mm:ss] timestamps, matching the house style
of Motivewave/1.Guide/MILK/mp4/captions/*.md so citations read the same way.

No audio is touched and no model is loaded — this is a pure text transform, so
it runs in seconds and can be re-run freely.

    uv run python transcripts_to_md.py --dry-run   # list what would be written
    uv run python transcripts_to_md.py             # write the .md files
"""

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path("/home/jerry/Projects/Trading")
LOG_PATH = Path(__file__).resolve().parent / "logs" / "transcripts_to_md.log"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}

# Paragraph rule, reverse-engineered from the six hand-made MILK caption .md
# files: close a paragraph once it spans 60s, or early on a real pause. These
# values reproduce three of those six files exactly and the rest within one
# paragraph.
MAX_PARAGRAPH_SECONDS = 60.0
PAUSE_BREAK_SECONDS = 3.5

CAVEAT = (
    "> Timestamps in `[mm:ss]` map back to the source video. Transcript is "
    "auto-generated and may contain mishearings (e.g. trading jargon, ticker names)."
)

TIMECODE_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)

log = logging.getLogger("transcripts_to_md")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, text) per cue, in file order."""
    segments = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if len(lines) < 3:
            continue
        match = TIMECODE_RE.search(lines[1])
        if not match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in match.groups())
        text = " ".join(lines[2:]).strip()
        if text:
            segments.append(
                (h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                 h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                 text)
            )
    return segments


def paragraphs(segments) -> list[tuple[float, str]]:
    """Group cues into (start, text) paragraphs."""
    out: list[tuple[float, list[str]]] = []
    start = prev_end = None
    for seg_start, seg_end, text in segments:
        new_para = (
            start is None
            or seg_start - prev_end >= PAUSE_BREAK_SECONDS
            or seg_end - start >= MAX_PARAGRAPH_SECONDS
        )
        if new_para:
            out.append((seg_start, [text]))
            start = seg_start
        else:
            out[-1][1].append(text)
        prev_end = seg_end
    return [(s, re.sub(r"\s+", " ", " ".join(parts)).strip()) for s, parts in out]


def clock(seconds: float) -> str:
    """h:mm:ss past the hour, mm:ss before — matches the MILK captions."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def render(srt: Path, model: str) -> str | None:
    segments = parse_srt(srt)
    if not segments:
        return None
    paras = paragraphs(segments)
    title = srt.stem.replace("_", " ")
    lines = [
        f"# {title}",
        "",
        f"- **Source:** `{srt.stem}` (Whisper auto-transcript, `{model}` model)",
        f"- **Total runtime:** {clock(segments[-1][1])}",
        f"- **Paragraph count:** {len(paras)}",
        "",
        CAVEAT,
        "",
        "---",
        "",
    ]
    lines += [f"**[{clock(start)}]** {text}\n" for start, text in paras]
    return "\n".join(lines)


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="directory to walk (default: %(default)s)")
    ap.add_argument("--model", default="turbo", help="model name to record in the header (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="rewrite .md files that already exist")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    if not args.root.is_dir():
        log.error("no such directory: %s", args.root)
        return 2

    videos = find_videos(args.root)
    log.info("=" * 72)
    log.info("found %d video(s) under %s", len(videos), args.root)

    written = skipped = no_srt = empty = 0
    for video in videos:
        srt = video.with_suffix(".srt")
        target = video.with_suffix(".md")

        if not srt.is_file():
            no_srt += 1
            log.warning("NO SRT   %s", video)
            continue
        if target.exists() and target.stat().st_size > 0 and not args.force:
            skipped += 1
            continue

        body = render(srt, args.model)
        if body is None:
            empty += 1
            log.warning("NO CUES  %s", srt)
            continue

        if args.dry_run:
            log.info("WOULD WRITE  %s", target)
        else:
            target.write_text(body, encoding="utf-8")
        written += 1

    log.info("-" * 72)
    log.info("videos %d · written %d · skipped %d · missing srt %d · empty srt %d%s",
             len(videos), written, skipped, no_srt, empty,
             "  (dry run: nothing written)" if args.dry_run else "")
    return 1 if (no_srt or empty) else 0


if __name__ == "__main__":
    sys.exit(main())

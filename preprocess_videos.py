"""
Video → JPEG frame extraction
==============================
Extracts frames from all videos in VIDEO_DIR at a fixed output FPS
and saves them as JPEG files under FRAMES_DIR.

Output layout:
  FRAMES_DIR/
    <video_stem>/
      000000.jpg
      000001.jpg
      ...

Run once before training:
  python preprocess_videos.py

By default uses config_p3 settings (games dataset).
Pass --config p2 to use config_p2 settings instead.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

FFMPEG  = r"C:\Users\oranc\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE = r"C:\Users\oranc\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["p2", "p3"], default="p3")
parser.add_argument("--fps",    type=float, default=6.0,
                    help="Output frame rate (default 6).")
parser.add_argument("--size",   type=int,   default=None,
                    help="Resize output frames to SIZE×SIZE pixels "
                         "(e.g. --size 64).  Defaults to original resolution. "
                         "Highly recommended: --size 64 makes JPEG loading "
                         "~100x faster during training.")
parser.add_argument("--quality", type=int, default=4,
                    help="JPEG quality scale (ffmpeg -q:v, 1=best 31=worst, default 4).")
parser.add_argument("--force", action="store_true",
                    help="Re-extract all videos even if frames already exist.")
args = parser.parse_args()

if args.config == "p2":
    import config_p2 as cfg
    FRAMES_DIR = f"frames_p2_{int(args.fps)}fps" + (f"_{args.size}px" if args.size else "")
else:
    import config_p3 as cfg
    FRAMES_DIR = f"frames_p3_{int(args.fps)}fps" + (f"_{args.size}px" if args.size else "")

VIDEO_DIR      = cfg.VIDEO_DIR
TRIM_START_SEC = cfg.TRIM_START_SEC
TRIM_END_SEC   = cfg.TRIM_END_SEC

_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def find_videos(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in _VIDEO_EXTS:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def get_duration(path: str) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract(video_path: str, out_dir: str, fps: float,
            trim_start: float, trim_end: float, quality: int):
    os.makedirs(out_dir, exist_ok=True)

    duration = get_duration(video_path)
    if duration <= trim_start + trim_end + 1:
        print(f"  SKIP (too short after trim): {video_path}")
        return 0

    end_time = duration - trim_end

    # ffmpeg: seek to trim_start, stop at end_time, extract at target fps
    scale = f"{args.size}:{args.size}" if args.size else "iw:ih"
    cmd = [
        FFMPEG, "-y",
        "-ss", str(trim_start),
        "-to", str(end_time),
        "-i", video_path,
        "-vf", f"fps={fps},scale={scale}",
        "-q:v", str(quality),
        "-f", "image2",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING ffmpeg error:\n{result.stderr[-500:]}")
        return 0

    n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    return n


def main():
    videos = find_videos(VIDEO_DIR)
    if not videos:
        print(f"No videos found in {VIDEO_DIR!r}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {VIDEO_DIR!r}")
    print(f"Output: {FRAMES_DIR!r}  |  {args.fps} fps  |  "
          f"trim {TRIM_START_SEC}s/{TRIM_END_SEC}s  |  JPEG quality {args.quality}")
    print()

    total = 0
    for video_path in videos:
        stem    = Path(video_path).stem
        out_dir = os.path.join(FRAMES_DIR, stem)

        # Skip if already extracted (unless --force)
        existing = len([f for f in os.listdir(out_dir)
                        if f.endswith(".jpg")]) if os.path.isdir(out_dir) else 0
        if existing > 0 and not args.force:
            print(f"  SKIP (already extracted {existing:,} frames): {stem}")
            total += existing
            continue

        print(f"Extracting: {stem}")
        n = extract(video_path, out_dir, args.fps,
                    TRIM_START_SEC, TRIM_END_SEC, args.quality)
        print(f"  -> {n:,} frames")
        total += n

    print(f"\nDone. Total frames: {total:,}")
    print(f"Update FRAMES_DIR in config to: {FRAMES_DIR!r}")


if __name__ == "__main__":
    main()

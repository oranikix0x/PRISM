"""
Remux game videos with faststart — no re-encode, just moves the moov atom
to the front of the file so decoders can build the keyframe index instantly.

Before: decord opens each video in ~5s (reads entire file for keyframe table)
After:  decord opens in < 0.5s (keyframe table at start of file)

Usage:
  python remux_faststart.py          # remuxes in-place (renames originals .bak)
  python remux_faststart.py --copy   # writes to videos/games_faststart/
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FFMPEG = r"C:\Users\oranc\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
VIDEO_DIR = r"C:\Projects\Personal\OIG\videos\games"
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

parser = argparse.ArgumentParser()
parser.add_argument("--copy", action="store_true",
                    help="Write to a separate directory instead of in-place.")
args = parser.parse_args()

videos = sorted([
    os.path.join(r, f)
    for r, _, fs in os.walk(VIDEO_DIR)
    for f in fs
    if Path(f).suffix.lower() in _VIDEO_EXTS
])

if not videos:
    print(f"No videos found in {VIDEO_DIR!r}")
    sys.exit(1)

print(f"Found {len(videos)} videos — remuxing with +faststart (no re-encode)...")
print()

total_saved = 0
for src in videos:
    name = Path(src).name[:70]
    if args.copy:
        out_dir = os.path.join(VIDEO_DIR + "_faststart")
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, Path(src).name)
for src in videos:
    name = Path(src).name[:70]
    if args.copy:
        out_dir = os.path.join(VIDEO_DIR + "_faststart")
        os.makedirs(out_dir, exist_ok=True)
        final_dst = os.path.join(out_dir, Path(src).name)
    else:
        final_dst = src   # overwrite in-place after temp write

    # Write to an ASCII temp path — FFmpeg's Windows binary uses ANSI file API
    # and fails on accented characters in the output path.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", src,
        "-c", "copy",
        "-movflags", "+faststart",
        tmp_path,
    ]
    print(f"  {name} ...", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED\n{r.stderr[-300:]}")
        os.remove(tmp_path)
        continue

    src_size = os.path.getsize(src)
    tmp_size = os.path.getsize(tmp_path)

    if not args.copy:
        bak = src + ".bak"
        os.rename(src, bak)
        os.rename(tmp_path, final_dst)
    else:
        os.rename(tmp_path, final_dst)

    print(f" done  ({src_size/1e6:.0f} MB -> {tmp_size/1e6:.0f} MB)")

print(f"\nAll done.")
if not args.copy:
    print(f"Original files renamed to .bak — delete them once training looks good.")

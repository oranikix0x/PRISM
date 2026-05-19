"""
Video frame dataset for Phase 2 / 3.

Returns consecutive quadruplets (frame_{t-2}, frame_{t-1}, frame_t, frame_{t+1})
from one of three backends (tried in this order):

  1. JPEG frames  — pre-extracted files from preprocess_videos.py (fastest)
  2. decord       — O(1) random seek on original video files (recommended)
  3. OpenCV       — last-resort fallback (slow H.264 random seeks)

  frame_{t-2}, frame_{t-1}  — context for step 1
  frame_t                   — step-1 prediction target
  frame_{t+1}               — step-2 prediction target
"""

import contextlib
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config_p2 as cfg

# ── Optional decord import (lazy — avoids slow GPU detection at import time) ──
_DECORD_AVAILABLE: bool | None = None   # None = not yet checked

def _check_decord() -> bool:
    global _DECORD_AVAILABLE
    if _DECORD_AVAILABLE is None:
        try:
            import decord  # noqa: F401
            _DECORD_AVAILABLE = True
        except ImportError:
            _DECORD_AVAILABLE = False
    return _DECORD_AVAILABLE

def _decord_imports():
    from decord import VideoReader, cpu as decord_cpu
    return VideoReader, decord_cpu

_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}

_RESIZE = T.Resize(
    (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


# ── Stderr suppression (OpenCV/FFmpeg fallback only) ─────────────────────────

@contextlib.contextmanager
def _quiet_stderr():
    """Temporarily redirect fd 2 → /dev/null (suppresses FFmpeg NAL warnings)."""
    try:
        devnull  = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except OSError:
        yield
    finally:
        try:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        except Exception:
            pass


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Silence FFmpeg stderr in DataLoader worker processes."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        pass


# ── Frame conversion helper ───────────────────────────────────────────────────

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (3, H, W) float32 [0,1], resized."""
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return _RESIZE(t).clamp(0.0, 1.0)


# ── Scene-cut detection ───────────────────────────────────────────────────────
# Threshold comes from config.SCENE_CUT_THRESHOLD (None / 0.0 = disabled).
# Mean absolute pixel difference above the threshold across any consecutive
# pair flags the tuple as a hard cut (teleport, loading screen, chapter jump).
_CUT_THRESHOLD: float | None = getattr(cfg, "SCENE_CUT_THRESHOLD", None) or None

def _is_scene_cut(*frames: torch.Tensor) -> bool:
    """Return True if any consecutive pair looks like a hard cut, or False if filter is off."""
    if not _CUT_THRESHOLD:
        return False
    for a, b in zip(frames, frames[1:]):
        if (a - b).abs().mean().item() > _CUT_THRESHOLD:
            return True
    return False


# ── Video helpers ─────────────────────────────────────────────────────────────

def _find_videos(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in _VIDEO_EXTS:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def _build_index_decord(
    videos:         list[str],
    frame_skip:     int,
    trim_start_sec: float = 0.0,
    trim_end_sec:   float = 0.0,
) -> list[tuple[str, int]]:
    """Build (video_path, t) index using decord for reliable frame counts."""
    VideoReader, decord_cpu = _decord_imports()
    index = []
    for path in videos:
        try:
            # Open at tiny size just to read frame count & fps — minimal memory
            vr  = VideoReader(path, ctx=decord_cpu(0), width=64, height=64,
                              num_threads=1)
            n   = len(vr)
            fps = vr.get_avg_fps() or 30.0
            del vr
        except Exception:
            continue
        t_start = max(2,     int(trim_start_sec * fps))
        t_end   = min(n - 2, n - 1 - int(trim_end_sec * fps))
        for t in range(t_start, t_end + 1, frame_skip):
            index.append((path, t))
    return index


def _build_index_cv2(
    videos:         list[str],
    frame_skip:     int,
    trim_start_sec: float = 0.0,
    trim_end_sec:   float = 0.0,
) -> list[tuple[str, int]]:
    """Build (video_path, t) index using OpenCV for frame counts."""
    index = []
    for path in videos:
        with _quiet_stderr():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        t_start = max(2,     int(trim_start_sec * fps))
        t_end   = min(n - 2, n - 1 - int(trim_end_sec * fps))
        for t in range(t_start, t_end + 1, frame_skip):
            index.append((path, t))
    return index


from collections import OrderedDict

_vr_cache: OrderedDict = OrderedDict()
_VR_CACHE_MAX = 8   # max simultaneously open VideoReaders (one per video is ideal)


def _get_vr(path: str, width: int = cfg.IMAGE_SIZE,
            height: int = cfg.IMAGE_SIZE) -> "VideoReader":
    """
    Return a cached VideoReader that decodes directly at (width, height).
    Decoding at target resolution skips full-res buffer allocation entirely —
    critical for 1080p sources where each raw frame is ~6 MB.
    """
    VideoReader, decord_cpu = _decord_imports()
    key = (path, width, height)
    if key in _vr_cache:
        _vr_cache.move_to_end(key)
        return _vr_cache[key]
    vr = VideoReader(path, ctx=decord_cpu(0), width=width, height=height,
                     num_threads=1)
    _vr_cache[key] = vr
    if len(_vr_cache) > _VR_CACHE_MAX:
        _vr_cache.popitem(last=False)   # evict LRU
    return vr


def _read_quad_decord(path: str, t: int):
    """
    Read frames t-2…t+1 using decord.
    Frames are decoded at cfg.IMAGE_SIZE directly — no full-res buffer needed.
    """
    try:
        vr     = _get_vr(path)
        batch  = vr.get_batch([t - 2, t - 1, t, t + 1]).asnumpy()  # (4,H,W,3) RGB
        # Already at target size; just convert to tensor and clamp
        return tuple(
            torch.from_numpy(batch[i]).permute(2, 0, 1).float().div_(255.0).clamp_(0.0, 1.0)
            for i in range(4)
        )
    except Exception:
        _vr_cache.pop((path, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), None)
        return None


def _read_quad_cv2(path: str, t: int):
    """Read frames t-2…t+1 using OpenCV (slow seek fallback)."""
    with _quiet_stderr():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t - 2)
            frames = []
            for _ in range(4):
                ok, f = cap.read()
                if not ok:
                    return None
                frames.append(_to_tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            return tuple(frames)
        finally:
            cap.release()


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoPairDataset(Dataset):
    """
    Each item is (frame_{t-2}, frame_{t-1}, frame_t, frame_{t+1}),
    all (3, H, W) float32 in [0, 1].

    Backend priority:
      1. JPEG frames  (frames_dir set and exists)
      2. decord       (installed — fast O(1) seeks on original videos)
      3. OpenCV       (fallback — slow, use only if decord unavailable)
    """

    def __init__(
        self,
        video_dir:      str        = cfg.VIDEO_DIR,
        frames_dir:     str | None = getattr(cfg, "FRAMES_DIR", None),
        frame_skip:     int        = cfg.FRAME_SKIP,
        max_pairs:      int | None = cfg.MAX_VIDEO_PAIRS,
        augment:        bool       = True,
        trim_start_sec: float      = cfg.TRIM_START_SEC,
        trim_end_sec:   float      = cfg.TRIM_END_SEC,
    ):
        self.augment  = augment
        self._backend = "jpeg"   # "jpeg" | "decord" | "cv2"

        # ── 1. JPEG frames ────────────────────────────────────────────────────
        if frames_dir and os.path.isdir(frames_dir):
            subdirs = sorted([
                os.path.join(frames_dir, d)
                for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d))
            ])
            index: list = []
            for subdir in subdirs:
                jpgs = sorted([f for f in os.listdir(subdir) if f.endswith(".jpg")])
                n = len(jpgs)
                for t in range(2, n - 1, frame_skip):
                    index.append((subdir, jpgs, t))
            if index:
                self._full_index = index
                self._max_items  = max_pairs
                self.reshuffle()
                print(f"VideoPairDataset [JPEG]: {len(self._full_index):,} total, "
                      f"{len(self.index):,} per epoch from {len(subdirs)} clips.")
                return

        # ── 2 & 3. Video file mode ────────────────────────────────────────────
        print(f"Scanning videos in {video_dir!r} ...")
        videos = _find_videos(video_dir)
        if not videos:
            raise FileNotFoundError(
                f"No video files found under {video_dir!r}.\n"
                f"Supported extensions: {sorted(_VIDEO_EXTS)}"
            )
        if trim_start_sec > 0 or trim_end_sec > 0:
            print(f"  Trim: skip first {trim_start_sec}s / last {trim_end_sec}s.")

        if _check_decord():
            self._backend = "decord"
            vid_index = _build_index_decord(videos, frame_skip,
                                            trim_start_sec, trim_end_sec)
            backend_label = "decord"
        else:
            self._backend = "cv2"
            vid_index = _build_index_cv2(videos, frame_skip,
                                         trim_start_sec, trim_end_sec)
            backend_label = "OpenCV (slow — pip install decord for fast seeks)"

        if not vid_index:
            raise RuntimeError("Found video files but could not read any frames.")
        self._full_index = vid_index
        self._max_items  = max_pairs
        self.reshuffle()
        print(f"VideoPairDataset [{backend_label}]: {len(self._full_index):,} total, "
              f"{len(self.index):,} per epoch from {len(videos)} videos.")

    def reshuffle(self) -> None:
        """Re-sample a fresh random subset of the full index for the next epoch."""
        if self._max_items and self._max_items < len(self._full_index):
            self.index = random.sample(self._full_index, self._max_items)
        else:
            self.index = random.sample(self._full_index, len(self._full_index))

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, idx: int):
        if self._backend == "jpeg":
            subdir, jpgs, t = self.index[idx]
            def read_jpg(name):
                path = os.path.join(subdir, name)
                # np.fromfile handles Unicode paths on Windows (cv2.imread can't).
                # Use IMREAD_REDUCED_COLOR_8 so libjpeg decodes at 1/8 resolution
                # natively — ~8× faster than full decode + Python resize for 1080p sources.
                raw = np.fromfile(path, dtype=np.uint8)
                if raw.size == 0:
                    return None
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                return _to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames = [read_jpg(jpgs[t - 2]), read_jpg(jpgs[t - 1]),
                      read_jpg(jpgs[t]),     read_jpg(jpgs[t + 1])]
            return None if None in frames else tuple(frames)
        elif self._backend == "decord":
            path, t = self.index[idx]
            return _read_quad_decord(path, t)
        else:
            path, t = self.index[idx]
            return _read_quad_cv2(path, t)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for _ in range(10):
            result = self._load(idx)
            if result is not None:
                break
            idx = random.randrange(len(self.index))
        else:
            dummy = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            return dummy, dummy, dummy, dummy

        frame_pp, frame_prev, frame_t, frame_t1 = result

        # Scene-cut filter: if any consecutive pair has large pixel diff it's
        # likely a hard cut.  Retry with a random sample rather than training
        # on a semantically meaningless (unrelated-frame) pair.
        if _is_scene_cut(frame_pp, frame_prev, frame_t, frame_t1):
            idx = random.randrange(len(self.index))
            result = self._load(idx)
            if result is None:
                dummy = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
                return dummy, dummy, dummy, dummy
            frame_pp, frame_prev, frame_t, frame_t1 = result

        return frame_pp, frame_prev, frame_t, frame_t1


# ── VideoSequenceDataset ──────────────────────────────────────────────────────

class VideoSequenceDataset(Dataset):
    """
    Returns a contiguous sequence of `seq_len` frames from a video clip.

    Each item is a tuple of `seq_len` tensors, each (3, H, W) float32 in [0,1].
    Used for Phase 3b multi-step unrolled training, where seq_len =
    P3B_UNROLL_STEPS + 2 (two ground-truth context frames + T prediction targets).

    Supports the same backends as VideoPairDataset (JPEG / decord / OpenCV).
    Within each returned sequence the frames are consecutive (stride-1 in
    whichever source the backend reads from).  The starting positions of
    sequences are sampled every `frame_skip` positions.

    DataLoader default collate_fn turns a batch of tuples-of-tensors into a
    tuple of (B, 3, H, W) tensors — one per time step — which the training
    loop can index directly as `frames[t]`.
    """

    def __init__(
        self,
        seq_len:        int,
        video_dir:      str        = cfg.VIDEO_DIR,
        frames_dir:     str | None = getattr(cfg, "FRAMES_DIR", None),
        frame_skip:     int        = 1,
        max_seqs:       int | None = None,
        augment:        bool       = True,
        trim_start_sec: float      = cfg.TRIM_START_SEC,
        trim_end_sec:   float      = cfg.TRIM_END_SEC,
    ):
        self.seq_len  = seq_len
        self.augment  = augment
        self._backend = "jpeg"

        # ── 1. JPEG frames ────────────────────────────────────────────────────
        if frames_dir and os.path.isdir(frames_dir):
            subdirs = sorted([
                os.path.join(frames_dir, d)
                for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d))
            ])
            index: list = []
            for subdir in subdirs:
                jpgs = sorted([f for f in os.listdir(subdir) if f.endswith(".jpg")])
                n    = len(jpgs)
                for t_start in range(0, n - seq_len + 1, frame_skip):
                    index.append((subdir, jpgs, t_start))
            if index:
                self._full_index = index
                self._max_items  = max_seqs
                self.reshuffle()
                print(f"VideoSequenceDataset [JPEG]: {len(self._full_index):,} total, "
                      f"{len(self.index):,} per epoch (len={seq_len}) from {len(subdirs)} clips.")
                return

        # ── 2 & 3. Video file mode ────────────────────────────────────────────
        print(f"Scanning videos in {video_dir!r} ...")
        videos = _find_videos(video_dir)
        if not videos:
            raise FileNotFoundError(
                f"No video files found under {video_dir!r}.\n"
                f"Supported extensions: {sorted(_VIDEO_EXTS)}"
            )
        if trim_start_sec > 0 or trim_end_sec > 0:
            print(f"  Trim: skip first {trim_start_sec}s / last {trim_end_sec}s.")

        if _check_decord():
            self._backend = "decord"
            vid_index = self._build_seq_index_decord(
                videos, seq_len, frame_skip, trim_start_sec, trim_end_sec)
            backend_label = "decord"
        else:
            self._backend = "cv2"
            vid_index = self._build_seq_index_cv2(
                videos, seq_len, frame_skip, trim_start_sec, trim_end_sec)
            backend_label = "OpenCV (slow — pip install decord for fast seeks)"

        if not vid_index:
            raise RuntimeError("Found video files but could not read any frames.")
        self._full_index = vid_index
        self._max_items  = max_seqs
        self.reshuffle()
        print(f"VideoSequenceDataset [{backend_label}]: {len(self._full_index):,} total, "
              f"{len(self.index):,} per epoch (len={seq_len}) from {len(videos)} videos.")

    # ── Index builders ────────────────────────────────────────────────────────

    @staticmethod
    def _build_seq_index_decord(
        videos:         list[str],
        seq_len:        int,
        frame_skip:     int,
        trim_start_sec: float,
        trim_end_sec:   float,
    ) -> list[tuple[str, int]]:
        VideoReader, decord_cpu = _decord_imports()
        index = []
        for path in videos:
            try:
                vr  = VideoReader(path, ctx=decord_cpu(0), width=64, height=64,
                                  num_threads=1)
                n   = len(vr)
                fps = vr.get_avg_fps() or 30.0
                del vr
            except Exception:
                continue
            t_start_min = int(trim_start_sec * fps)
            t_end_max   = n - seq_len - int(trim_end_sec * fps)
            for t in range(t_start_min, t_end_max + 1, frame_skip):
                index.append((path, t))
        return index

    @staticmethod
    def _build_seq_index_cv2(
        videos:         list[str],
        seq_len:        int,
        frame_skip:     int,
        trim_start_sec: float,
        trim_end_sec:   float,
    ) -> list[tuple[str, int]]:
        index = []
        for path in videos:
            with _quiet_stderr():
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            t_start_min = int(trim_start_sec * fps)
            t_end_max   = n - seq_len - int(trim_end_sec * fps)
            for t in range(t_start_min, t_end_max + 1, frame_skip):
                index.append((path, t))
        return index

    def reshuffle(self) -> None:
        """Re-sample a fresh random subset of the full index for the next epoch."""
        if self._max_items and self._max_items < len(self._full_index):
            self.index = random.sample(self._full_index, self._max_items)
        else:
            self.index = random.sample(self._full_index, len(self._full_index))

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, idx: int) -> tuple | None:
        if self._backend == "jpeg":
            subdir, jpgs, t_start = self.index[idx]
            frames = []
            for i in range(self.seq_len):
                path = os.path.join(subdir, jpgs[t_start + i])
                raw  = np.fromfile(path, dtype=np.uint8)
                if raw.size == 0:
                    return None
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                frames.append(_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            return tuple(frames)

        elif self._backend == "decord":
            path, t_start = self.index[idx]
            try:
                vr    = _get_vr(path)
                batch = vr.get_batch(list(range(t_start, t_start + self.seq_len))).asnumpy()
                return tuple(
                    torch.from_numpy(batch[i]).permute(2, 0, 1).float().div_(255.0).clamp_(0.0, 1.0)
                    for i in range(self.seq_len)
                )
            except Exception:
                _vr_cache.pop((path, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), None)
                return None

        else:  # cv2
            path, t_start = self.index[idx]
            with _quiet_stderr():
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    return None
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, t_start)
                    frames = []
                    for _ in range(self.seq_len):
                        ok, f = cap.read()
                        if not ok:
                            return None
                        frames.append(_to_tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    return tuple(frames)
                finally:
                    cap.release()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple:
        for _ in range(10):
            result = self._load(idx)
            if result is not None:
                break
            idx = random.randrange(len(self.index))
        else:
            dummy = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            return tuple(dummy for _ in range(self.seq_len))

        # Scene-cut filter: skip sequences that contain a hard cut.
        if _is_scene_cut(*result):
            idx = random.randrange(len(self.index))
            result = self._load(idx)
            if result is None:
                dummy = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
                return tuple(dummy for _ in range(self.seq_len))

        return result


# ── PermutedActionDataset (Phase 4) ──────────────────────────────────────────

class PermutedActionDataset(VideoSequenceDataset):
    """
    Thin wrapper around VideoSequenceDataset for Phase 4 two-agent training.

    Returns the same contiguous frame sequences as VideoSequenceDataset.
    The permutation of action codes is performed inside train_p4.py after
    LAM infers action codes from consecutive real frame pairs — this dataset
    simply provides the raw frame windows.

    Use VideoSequenceDataset directly if you prefer, or use this class for
    readability / documentation purposes.  The seq_len should be P4_SEQ_LEN+1
    (one seed frame + P4_SEQ_LEN prediction targets).
    """
    pass


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    ds = VideoPairDataset()
    t0 = time.time()
    fpp, fp, ft, ft1 = ds[0]
    print(f"First item loaded in {time.time()-t0:.2f}s  "
          f"backend={ds._backend}  shape={ft.shape}")
    t0 = time.time()
    for i in range(10):
        ds[random.randrange(len(ds))]
    print(f"10 random items: {time.time()-t0:.2f}s avg")

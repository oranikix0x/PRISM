"""
Flickr30k image dataset — returns only images (no captions needed here).

Images are centre-cropped to a square, resized to IMAGE_SIZE × IMAGE_SIZE,
and normalised to [0, 1] float32 tensors.
"""

import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _center_square_crop(img: Image.Image) -> Image.Image:
    side = min(img.size)
    left = (img.width  - side) // 2
    top  = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


_TRANSFORM = T.Compose([
    T.Lambda(_center_square_crop),
    T.Resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    ),
    T.ToTensor(),   # → float32 in [0, 1], shape (3, H, W)
])


class Flickr30kImages(Dataset):
    """
    Loads every image file from a flat directory.

    Args:
        images_dir: path to folder containing .jpg / .png / … files.
        max_images: optional cap on dataset size (sampled uniformly at random).
        augment:    if True, applies random horizontal flip during training.
    """

    def __init__(
        self,
        images_dir: str = config.FLICKR_DIR,
        max_images: int | None = config.MAX_IMAGES,
        augment: bool = True,
    ):
        paths = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in _EXTENSIONS
        ]
        if not paths:
            raise FileNotFoundError(f"No images found in {images_dir!r}")

        self._all_paths  = paths
        self._max_images = max_images
        self.augment     = augment
        self._hflip      = T.RandomHorizontalFlip(p=0.5)
        self.reshuffle()
        print(f"Flickr30kImages: {len(self._all_paths)} total, "
              f"{len(self.paths)} per epoch  (dir={images_dir!r})")

    def reshuffle(self) -> None:
        """Re-sample a fresh random subset for the next epoch."""
        if self._max_images is not None and self._max_images < len(self._all_paths):
            self.paths = random.sample(self._all_paths, self._max_images)
        else:
            self.paths = random.sample(self._all_paths, len(self._all_paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img    = Image.open(self.paths[idx]).convert("RGB")
        tensor = _TRANSFORM(img)
        if self.augment:
            tensor = self._hflip(tensor)
        return tensor

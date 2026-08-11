"""Semantic segmentation dataset utilities.

This module provides a simple PyTorch Dataset for folder-based semantic
segmentation data laid out as:

    root/
      train/
        images/
        masks/
      valid/
        images/
        masks/

Images are expected to be `.jpg` files and masks are expected to be `.png`
files with the same file stem as the image.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class SemanticSegmentationDataset(Dataset):
    """Folder-based semantic segmentation dataset.

    Args:
        root_dir: Dataset root containing split subfolders like `train` and
            `valid`.
        split: Which split to load, for example `train` or `valid`.
        transforms: Optional callable accepting `(image, mask)` and returning
            the transformed `(image, mask)`.
        image_suffix: File suffix used for images. Defaults to `.jpg`.
        mask_suffix: File suffix used for masks. Defaults to `.png`.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms: Optional[Callable[[tv_tensors.Image, tv_tensors.Mask], Tuple[tv_tensors.Image, tv_tensors.Mask]]] = None,
        image_suffix: str = ".png",
        mask_suffix: str = ".png",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        image_paths = sorted(self.images_dir.glob(f"*{self.image_suffix}"))

        if not image_paths:
            raise ValueError(f"No image files found in: {self.images_dir}")

        for image_path in image_paths:
            mask_path = self.masks_dir / f"{image_path.stem}{self.mask_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for image {image_path.name}: expected {mask_path}"
                )
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        image_path, mask_path = self.samples[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if image.size != mask.size:
            raise ValueError(
                f"Image and mask size mismatch for {image_path.name}: "
                f"image={image.size}, mask={mask.size}"
            )

        image_tensor = self._to_tv_image(image)
        mask_tensor = self._to_tv_mask(mask)

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        # Keep the mask in its native indexed/palette form so class IDs stay
        # intact when we convert it to a tensor.
        with Image.open(path) as mask:
            return mask.copy()

    def _to_tv_image(self, image: Image.Image) -> tv_tensors.Image:
        return tv_tensors.Image(F.pil_to_tensor(image))

    def _to_tv_mask(self, mask: Image.Image) -> tv_tensors.Mask:
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return tv_tensors.Mask(mask_tensor)

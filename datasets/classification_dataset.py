from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


DEFAULT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageClassificationDataset(Dataset):
    """Folder-based image classification dataset.

    Expected layout:

        root/
          train/
            class1/
            class2/
          eval/
            class1/
            class2/
          test/
            class1/
            class2/

    Class labels are inferred from the class folder names unless `classes` is
    provided. When reusing the dataset across splits, pass `classes` from the
    training dataset so label indices stay consistent.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transforms: Callable[[tv_tensors.Image], torch.Tensor] | None = None,
        classes: Sequence[str] | None = None,
        image_suffixes: Sequence[str] = DEFAULT_IMAGE_SUFFIXES,
        allow_missing_classes: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.image_suffixes = tuple(suffix.lower() for suffix in image_suffixes)
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        if classes is None:
            self.classes = sorted(
                [entry.name for entry in self.split_dir.iterdir() if entry.is_dir()]
            )
        else:
            self.classes = list(classes)

        if not self.classes:
            raise ValueError(f"No class folders found in: {self.split_dir}")

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                if allow_missing_classes:
                    continue
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in self.image_suffixes
            )
            for image_path in image_paths:
                self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No images found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, torch.Tensor]:
        image_path, label = self.samples[idx]
        image = self._load_image(image_path)
        image_tensor = self._to_tv_image(image)

        if self.transforms is not None:
            image_tensor = self.transforms(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _to_tv_image(self, image: Image.Image) -> tv_tensors.Image:
        return tv_tensors.Image(F.pil_to_tensor(image))


class ImageBinaryClassificationDataset(ImageClassificationDataset):
    """Folder-based binary image classification dataset.

    Expected layout:

        root/
          train/
            negative_class/
            positive_class/
          eval/
            negative_class/
            positive_class/
          test/
            negative_class/
            positive_class/

    The class folders are inferred from the split directory unless `classes`
    is provided. Exactly two classes must exist in each split.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transforms: Callable[[tv_tensors.Image], torch.Tensor] | None = None,
        classes: Sequence[str] | None = None,
        image_suffixes: Sequence[str] = DEFAULT_IMAGE_SUFFIXES,
        allow_missing_classes: bool = False,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            split=split,
            transforms=transforms,
            classes=classes,
            image_suffixes=image_suffixes,
            allow_missing_classes=allow_missing_classes,
        )

        if len(self.classes) != 2:
            raise ValueError(
                f"Binary classification requires exactly two class folders in {self.split_dir}, "
                f"got {len(self.classes)}: {self.classes}"
            )

        self.negative_class_name = self.classes[0]
        self.positive_class_name = self.classes[1]

"""Detection dataset module for COCO-style object detection data.

This module provides a modular PyTorch Dataset implementation that loads images
with PIL and returns image/target pairs in the format expected by common
PyTorch detection training loops.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def detection_collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Collate function for DataLoader using detection targets.

    Returns:
        Tuple of images and targets. Each target is a dict.
    """
    images, targets = zip(*batch)
    return images, targets


class CocoDetectionDataset(Dataset):
    """COCO-style detection dataset.

    Args:
        images_root: Root directory containing image files.
        annotation_path: Path to a COCO annotation JSON file.
        transforms: Optional callable(image, target) -> (image, target).
        use_coco_category_ids: If True, preserve original category IDs from the
            annotation file. If False, map categories to consecutive labels
            starting at 1.
    """

    def __init__(
        self,
        images_root: str,
        annotation_path: str,
        transforms: Optional[Callable[[Image.Image, Dict[str, Tensor]], Tuple[Image.Image, Dict[str, Tensor]]]] = None,
        use_coco_category_ids: bool = False,
    ) -> None:
        self.images_root = Path(images_root)
        self.annotation_path = Path(annotation_path)
        self.transforms = transforms
        self.use_coco_category_ids = use_coco_category_ids

        with self.annotation_path.open("r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.images = coco_data.get("images", [])
        self.annotations = coco_data.get("annotations", [])
        self.categories = coco_data.get("categories", [])

        if not self.images:
            raise ValueError(f"No images found in annotation file: {annotation_path}")

        self.image_id_to_info: Dict[int, Dict[str, Any]] = {
            image["id"]: image for image in self.images
        }
        self.image_ids: List[int] = list(self.image_id_to_info.keys())

        self.image_id_to_annotations: Dict[int, List[Dict[str, Any]]] = {
            image_id: [] for image_id in self.image_ids
        }
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id].append(ann)

        self.category_id_to_label: Dict[int, int] = {}
        if self.use_coco_category_ids:
            for category in self.categories:
                self.category_id_to_label[category["id"]] = category["id"]
        else:
            sorted_categories = sorted(self.categories, key=lambda item: item["id"])
            for index, category in enumerate(sorted_categories, start=1):
                self.category_id_to_label[category["id"]] = index

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[tv_tensors.Image, Dict[str, Tensor]]:
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]
        image_path = self.images_root / image_info["file_name"]

        image = self._load_image(image_path)
        image = self._to_tv_image(image)
        target = self._build_target(image_id, image_info, image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _build_target(self, image_id: int, image_info: Dict[str, Any], image: tv_tensors.Image) -> Dict[str, Tensor]:
        annotations = self.image_id_to_annotations.get(image_id, [])

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        areas_list: List[float] = []
        iscrowd_list: List[int] = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes_list.append([x, y, x + w, y + h])
            labels_list.append(self.category_id_to_label.get(ann["category_id"], ann["category_id"]))
            areas_list.append(float(ann.get("area", w * h)))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            area = torch.tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, Tensor] = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=tuple(F.get_size(image)),
            ),
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([image_info.get("height", 0), image_info.get("width", 0)], dtype=torch.int64),
            "size": torch.tensor([image_info.get("height", 0), image_info.get("width", 0)], dtype=torch.int64),
        }

        return target

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")

    def _to_tv_image(self, image: Image.Image) -> tv_tensors.Image:
        tensor_image = F.pil_to_tensor(image)
        return tv_tensors.Image(tensor_image)

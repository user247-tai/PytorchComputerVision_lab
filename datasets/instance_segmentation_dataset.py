"""COCO-style instance segmentation dataset utilities.

This module loads COCO instance segmentation annotations and returns image/target
pairs suitable for Mask R-CNN style training loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from .detection_dataset import detection_collate_fn


instance_collate_fn = detection_collate_fn


class COCOInstanceDataset(Dataset):
    """COCO-style instance segmentation dataset.

    Args:
        images_root: Root directory containing image files.
        annotation_path: Path to a COCO annotation JSON file.
        transforms: Optional callable(image, target) -> (image, target).
        use_coco_category_ids: If True, preserve original COCO category IDs.
            If False, map categories to consecutive labels starting at 1.
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
        target = self._build_target(image_id=image_id, image_info=image_info, image=image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _build_target(self, image_id: int, image_info: Dict[str, Any], image: tv_tensors.Image) -> Dict[str, Tensor]:
        annotations = self.image_id_to_annotations.get(image_id, [])
        height, width = int(image.shape[-2]), int(image.shape[-1])

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        areas_list: List[float] = []
        iscrowd_list: List[int] = []
        masks_list: List[torch.Tensor] = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes_list.append([x, y, x + w, y + h])
            labels_list.append(self.category_id_to_label.get(ann["category_id"], ann["category_id"]))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

            mask = self._annotation_to_mask(ann.get("segmentation", []), height, width)
            masks_list.append(mask)
            areas_list.append(float(ann.get("area", mask.sum().item())))

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            area = torch.tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd_list, dtype=torch.int64)
            masks = torch.stack(masks_list, dim=0).to(torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

        target: Dict[str, Tensor] = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=tuple(F.get_size(image)),
            ),
            "labels": labels,
            "masks": tv_tensors.Mask(masks),
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([image_info.get("height", 0), image_info.get("width", 0)], dtype=torch.int64),
            "size": torch.tensor([image_info.get("height", 0), image_info.get("width", 0)], dtype=torch.int64),
        }

        return target

    def _decode_coco_rle_counts(self, counts: str | bytes) -> list[int]:
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")

        decoded_counts: list[int] = []
        p = 0
        n = len(counts)

        while p < n:
            x = 0
            k = 0
            more = True

            while more:
                c = ord(counts[p]) - 48
                p += 1
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)

            if len(decoded_counts) > 2:
                x += decoded_counts[-2]

            decoded_counts.append(int(x))

        return decoded_counts

    def _rle_counts_to_mask(self, counts: list[int], height: int, width: int) -> torch.Tensor:
        total_pixels = height * width
        flat_mask = np.zeros(total_pixels, dtype=np.uint8)

        idx = 0
        value = 0
        for run_length in counts:
            run_length = int(run_length)
            if run_length < 0:
                raise ValueError(f"Invalid RLE run length: {run_length}")
            if run_length > 0 and value == 1:
                flat_mask[idx : idx + run_length] = 1
            idx += run_length
            value ^= 1

        if idx > total_pixels:
            raise ValueError(
                f"RLE decodes to more pixels than the mask size: {idx} > {total_pixels}"
            )

        return torch.from_numpy(flat_mask.reshape((height, width), order="F"))

    def _annotation_to_mask(self, segmentation: Any, height: int, width: int) -> torch.Tensor:
        if not segmentation:
            return torch.zeros((height, width), dtype=torch.uint8)

        if isinstance(segmentation, dict):
            counts = segmentation.get("counts")
            if counts is None:
                raise ValueError("COCO RLE segmentation dict must contain a 'counts' field")

            if isinstance(counts, list):
                rle_counts = [int(v) for v in counts]
            elif isinstance(counts, (str, bytes)):
                rle_counts = self._decode_coco_rle_counts(counts)
            else:
                raise TypeError(
                    f"Unsupported RLE counts type: {type(counts)}. Expected list, str, or bytes."
                )

            return self._rle_counts_to_mask(rle_counts, height, width)

        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            if not polygon:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if points.shape[0] < 3:
                continue
            pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)

        return torch.from_numpy(mask)

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")

    def _to_tv_image(self, image: Image.Image) -> tv_tensors.Image:
        tensor_image = F.pil_to_tensor(image)
        return tv_tensors.Image(tensor_image)

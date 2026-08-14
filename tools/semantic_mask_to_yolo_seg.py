"""Convert indexed semantic masks into YOLO segmentation labels.

YOLO segmentation training expects one text file per image where each line is:

    class_id x1 y1 x2 y2 x3 y3 ...

with coordinates normalized to [0, 1].

This script treats each connected component of each semantic class as one
polygon instance. That is the closest match to YOLO-seg training, even though
your source data is semantic masks rather than instance masks.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = [f"object_{idx}" for idx in range(1, 12)]
NUM_CLASSES = len(CLASS_NAMES)
BACKGROUND_LABEL = 0


def load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        return np.array(mask)


def find_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted([*images_dir.glob("*.png"), *images_dir.glob("*.jpg"), *images_dir.glob("*.jpeg")])
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    mask_by_stem = {p.stem: p for p in masks_dir.glob("*.png")}
    if not mask_by_stem:
        raise ValueError(f"No mask files found in {masks_dir}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image {image_path.name}")
        pairs.append((image_path, mask_path))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    if not pairs:
        return [], []

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0.0 and val_count == 0 and len(shuffled) > 1:
        val_count = 1
    if val_count >= len(shuffled):
        val_count = max(len(shuffled) - 1, 0)

    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]
    return train_pairs, val_pairs


def contour_to_yolo_polygon(contour: np.ndarray, width: int, height: int, epsilon_ratio: float = 0.002) -> list[float]:
    arc_len = cv2.arcLength(contour, True)
    epsilon = max(1.0, epsilon_ratio * arc_len)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx.shape[0] < 3:
        return []

    approx = approx.reshape(-1, 2)
    coords: list[float] = []
    for x, y in approx:
        x_norm = float(np.clip(x / width, 0.0, 1.0))
        y_norm = float(np.clip(y / height, 0.0, 1.0))
        coords.extend([x_norm, y_norm])
    return coords if len(coords) >= 6 else []


def mask_class_to_yolo_lines(mask: np.ndarray, class_label: int, width: int, height: int, min_area: float = 1.0) -> list[str]:
    binary = (mask == class_label).astype(np.uint8)
    if binary.max() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []

    yolo_class_id = class_label - 1
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        polygon = contour_to_yolo_polygon(contour, width, height)
        if not polygon:
            continue
        coords_text = " ".join(f"{value:.6f}" for value in polygon)
        lines.append(f"{yolo_class_id} {coords_text}")

    return lines


def convert_split(
    split_name: str,
    pairs: Iterable[tuple[Path, Path]],
    output_root: Path,
) -> int:
    split_images_dir = output_root / split_name / "images"
    split_labels_dir = output_root / split_name / "labels"
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path, mask_path in tqdm(list(pairs), desc=f"Converting {split_name}"):
        image = Image.open(image_path).convert("RGB")
        mask = load_mask(mask_path)

        width, height = image.size
        if mask.shape[0] != height or mask.shape[1] != width:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: image={image.size}, mask={mask.shape[::-1]}"
            )

        label_lines: list[str] = []
        unique_labels = sorted(int(v) for v in np.unique(mask).tolist() if int(v) != BACKGROUND_LABEL)
        for class_label in unique_labels:
            if class_label < 1 or class_label > NUM_CLASSES:
                raise ValueError(
                    f"Found label {class_label} in {mask_path.name}, but expected labels 0..{NUM_CLASSES}"
                )
            label_lines.extend(mask_class_to_yolo_lines(mask, class_label, width, height))

        shutil.copy2(image_path, split_images_dir / image_path.name)
        label_path = split_labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        count += 1

    return count


def write_data_yaml(output_root: Path) -> None:
    data_yaml = output_root / "data.yaml"
    yaml_text = f"""path: {output_root}
train: train/images
val: val/images
test: test/images

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    data_yaml.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert indexed semantic masks to YOLO segmentation labels.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/home/server/computer_vision_lab/data/segmentation/semantic/dataset1"),
        help="Source semantic segmentation dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/server/computer_vision_lab/data/segmentation/semantic/dataset1_yolo"),
        help="Output YOLO-seg dataset root.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the source train split to reserve for YOLO val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split.",
    )
    args = parser.parse_args()

    train_images_dir = args.src_root / "train" / "images"
    train_masks_dir = args.src_root / "train" / "masks"
    test_images_dir = args.src_root / "test" / "images"
    test_masks_dir = args.src_root / "test" / "masks"

    train_pairs = find_image_mask_pairs(train_images_dir, train_masks_dir)
    test_pairs = find_image_mask_pairs(test_images_dir, test_masks_dir)
    train_pairs, val_pairs = split_pairs(train_pairs, args.val_ratio, args.seed)

    print(f"Source train images: {len(train_pairs) + len(val_pairs)}")
    print(f"YOLO train images:   {len(train_pairs)}")
    print(f"YOLO val images:     {len(val_pairs)}")
    print(f"YOLO test images:    {len(test_pairs)}")
    print(f"Classes: {CLASS_NAMES}")

    train_count = convert_split("train", train_pairs, args.output_root)
    val_count = convert_split("val", val_pairs, args.output_root)
    test_count = convert_split("test", test_pairs, args.output_root)
    write_data_yaml(args.output_root)

    print("Done.")
    print(f"Wrote YOLO-seg dataset to: {args.output_root}")
    print(f"Converted counts: train={train_count}, val={val_count}, test={test_count}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()

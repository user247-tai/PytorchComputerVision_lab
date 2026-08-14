from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
]


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def load_instance_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        return np.array(mask, dtype=np.uint8)


def mask_to_polygons(binary_mask: np.ndarray) -> list[list[float]]:
    """Convert a binary mask into COCO-style polygon segmentation."""
    binary_mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        polygon = contour.flatten().astype(float).tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def instance_mask_to_annotations(
    mask: np.ndarray,
    image_id: int,
    start_annotation_id: int,
    category_id: int = 1,
) -> tuple[list[dict], int]:
    annotations: list[dict] = []
    annotation_id = start_annotation_id

    for instance_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        binary_mask = (mask == instance_id).astype(np.uint8)
        area = int(binary_mask.sum())
        if area == 0:
            continue

        ys, xs = np.where(binary_mask > 0)
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]

        segmentation = mask_to_polygons(binary_mask)

        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": float(area),
                "segmentation": segmentation,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    return annotations, annotation_id


def split_samples(
    samples: list[tuple[Path, Path]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(round(n * train_ratio))
    valid_end = train_end + int(round(n * valid_ratio))

    train_split = shuffled[:train_end]
    valid_split = shuffled[train_end:valid_end]
    test_split = shuffled[valid_end:]

    return {"train": train_split, "valid": valid_split, "test": test_split}


def collect_samples(input_root: Path) -> list[tuple[Path, Path]]:
    images_dir = input_root / "PNGImages"
    masks_dir = input_root / "PedMasks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}: expected {mask_path}")
        samples.append((image_path, mask_path))

    if not samples:
        raise ValueError(f"No images found in {images_dir}")

    return samples


def build_coco_split(
    split_name: str,
    samples: list[tuple[Path, Path]],
    output_root: Path,
    categories: list[dict],
    copy_images: bool = True,
) -> None:
    images_dir = output_root / split_name
    annotations_dir = output_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "PennFudanPed converted to COCO instance segmentation format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 1
    annotation_id = 1

    for image_path, mask_path in samples:
        width, height = load_image_size(image_path)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        mask = load_instance_mask(mask_path)
        annotations, annotation_id = instance_mask_to_annotations(
            mask=mask,
            image_id=image_id,
            start_annotation_id=annotation_id,
            category_id=1,
        )
        coco["annotations"].extend(annotations)

        if copy_images:
            shutil.copy2(image_path, images_dir / image_path.name)

        image_id += 1

    out_json_path = annotations_dir / f"instances_{split_name}.json"
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(
        f"Saved {split_name}: {len(samples)} images, {len(coco['annotations'])} annotations -> {out_json_path}"
    )


def convert_pennfudan_to_coco(
    input_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_images: bool = True,
) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)

    samples = collect_samples(input_root)
    splits = split_samples(samples, train_ratio, valid_ratio, test_ratio, seed)

    for split_name in ("train", "valid", "test"):
        build_coco_split(
            split_name=split_name,
            samples=splits[split_name],
            output_root=output_root,
            categories=DEFAULT_CATEGORIES,
            copy_images=copy_images,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PennFudanPed into COCO instance segmentation format with masks."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="/home/server/computer_vision_lab/data/segmentation/instance/PennFudanPed",
        help="Path to the PennFudanPed dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/server/computer_vision_lab/data/detection/pennfudan_coco",
        help="Where to write the COCO dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy image files into the COCO output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_pennfudan_to_coco(
        input_root=args.input_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    main()

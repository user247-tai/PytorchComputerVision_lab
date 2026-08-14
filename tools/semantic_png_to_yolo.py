"""Prepare a YOLO semantic-segmentation dataset from PNG mask labels.

This preserves the mask images instead of converting them to .txt polygon labels.
It matches the Ultralytics semantic dataset layout:

    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/

The masks are copied as-is, so each mask remains a single-channel indexed PNG.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ["background"] + [f"object_{idx}" for idx in range(1, 12)]
NUM_CLASSES = len(CLASS_NAMES)


def find_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted([
        *images_dir.glob("*.png"),
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
    ])
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    mask_by_stem = {p.stem: p for p in masks_dir.glob("*.png")}
    if not mask_by_stem:
        raise ValueError(f"No PNG masks found in {masks_dir}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image {image_path.name}")
        pairs.append((image_path, mask_path))
    return pairs


def split_train_val(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

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


def copy_split(split_name: str, pairs: list[tuple[Path, Path]], output_root: Path) -> int:
    images_out = output_root / "images" / split_name
    masks_out = output_root / "masks" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in tqdm(pairs, desc=f"Copying {split_name}"):
        with Image.open(mask_path) as mask:
            mask_mode = mask.mode
            if mask_mode not in {"L", "P", "I;16", "I"}:
                # Preserve the raw values by copying, but flag unusual formats early.
                print(f"Warning: {mask_path.name} has mode {mask_mode}; expected indexed/grayscale mask")
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(mask_path, masks_out / mask_path.name)

    return len(pairs)


def write_data_yaml(output_root: Path) -> None:
    data_yaml = output_root / "data.yaml"
    yaml_text = f"""path: {output_root}
train: images/train
val: images/val
test: images/test
masks_dir: masks

names:
"""
    for idx, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {idx}: {name}\n"
    data_yaml.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a YOLO semantic dataset from PNG masks.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/home/server/computer_vision_lab/data/segmentation/semantic/dataset1"),
        help="Source semantic segmentation dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/server/computer_vision_lab/data/segmentation/semantic/dataset1_yolo_semantic"),
        help="Output YOLO semantic dataset root.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the source train split to use as val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    args = parser.parse_args()

    train_images_dir = args.src_root / "train" / "images"
    train_masks_dir = args.src_root / "train" / "masks"
    test_images_dir = args.src_root / "test" / "images"
    test_masks_dir = args.src_root / "test" / "masks"

    train_pairs = find_pairs(train_images_dir, train_masks_dir)
    test_pairs = find_pairs(test_images_dir, test_masks_dir)
    train_pairs, val_pairs = split_train_val(train_pairs, args.val_ratio, args.seed)

    print(f"Class names: {CLASS_NAMES}")
    print(f"Source train images: {len(train_pairs) + len(val_pairs)}")
    print(f"YOLO train images:   {len(train_pairs)}")
    print(f"YOLO val images:     {len(val_pairs)}")
    print(f"YOLO test images:    {len(test_pairs)}")

    train_count = copy_split("train", train_pairs, args.output_root)
    val_count = copy_split("val", val_pairs, args.output_root)
    test_count = copy_split("test", test_pairs, args.output_root)
    write_data_yaml(args.output_root)

    print("Done.")
    print(f"Wrote YOLO semantic dataset to: {args.output_root}")
    print(f"Converted counts: train={train_count}, val={val_count}, test={test_count}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()

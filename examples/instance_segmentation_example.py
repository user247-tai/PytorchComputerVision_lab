from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToPureTensor

from datasets import COCOInstanceDataset, instance_collate_fn
from examples.common import device_from_arg, load_weights_into_model
from vision_tasks import InstanceSegmentation


DEFAULT_ROOT = Path("data/segmentation/instance/pennfudan_coco")
DEFAULT_WEIGHTS = Path("maskrcnn_pennfudan_cc.pt")
DEFAULT_ONNX = Path("instance_pt.onnx")

DEFAULT_LABEL_NAMES = {1: "pedestrian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instance segmentation task example")
    parser.add_argument("--mode", choices=["train", "valid", "test", "export", "all"], default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="sgd", help="Optimizer name: sgd, adam, adamw, rmsprop")
    parser.add_argument("--checkpoint", default="checkpoints/maskrcnn_pennfudan.pt")
    parser.add_argument("--trained-model-dir", default="trained_models")
    parser.add_argument("--trained-model-name", default="instance_segmentation_trained.pt")
    parser.add_argument("--weights", default="", help="Optional path to pretrained weights. Leave empty to train from scratch.")
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX))
    parser.add_argument("--output-dir", default="predictions_instance")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=12)
    return parser.parse_args()


def _train_transforms():
    return Compose([RandomHorizontalFlip(0.5), ToDtype(torch.float32, scale=True), ToPureTensor()])


def _eval_transforms():
    return Compose([ToDtype(torch.float32, scale=True), ToPureTensor()])


def build_datasets(root: Path, train_split: str, eval_split: str, test_split: str):
    train_dataset = COCOInstanceDataset(
        images_root=root / train_split,
        annotation_path=root / "annotations" / f"instances_{train_split}.json",
        transforms=_train_transforms(),
    )
    eval_dataset = COCOInstanceDataset(
        images_root=root / eval_split,
        annotation_path=root / "annotations" / f"instances_{eval_split}.json",
        transforms=_eval_transforms(),
    )
    test_dataset = COCOInstanceDataset(
        images_root=root / test_split,
        annotation_path=root / "annotations" / f"instances_{test_split}.json",
        transforms=_eval_transforms(),
    )
    return train_dataset, eval_dataset, test_dataset


def build_loaders(args: argparse.Namespace):
    root = Path(args.root)
    train_dataset, eval_dataset, test_dataset = build_datasets(root, args.train_split, args.eval_split, args.test_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=instance_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=instance_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=instance_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, test_loader


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    task = InstanceSegmentation(device=device, label_names=DEFAULT_LABEL_NAMES)
    task.build_model(num_classes=2, freeze_backbone=False, pretrained_backbone=True)

    if args.weights:
        load_weights_into_model(task.model, args.weights)

    train_loader, eval_loader, test_loader = build_loaders(args)
    trained_model_path = Path(args.trained_model_dir) / args.trained_model_name

    if args.mode in {"train", "all"}:
        task.train_model(
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_path=args.checkpoint,
            trained_model_path=trained_model_path,
            eval_loader=eval_loader,
            optimizer_name=args.optimizer,
        )

    if args.mode in {"valid", "all"}:
        task.evaluate_model(eval_loader=eval_loader, device=device)

    if args.mode in {"test", "all"}:
        task.test_model(
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            max_images=args.max_images,
        )

    if args.mode in {"export", "all"}:
        task.export_onnx(
            onnx_path=args.onnx_path,
            state_dict_path=args.weights if args.weights else None,
            num_classes=2,
        )


if __name__ == "__main__":
    main()

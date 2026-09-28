from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ray import tune
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage

from datasets import CocoDetectionDataset, detection_collate_fn
from examples.common import device_from_arg, load_weights_into_model
from vision_tasks import Detection


DEFAULT_ROOT = Path("data/detection/fire_smoke_coco")
DEFAULT_WEIGHTS = Path("fire_smoke.pt")
DEFAULT_ONNX = Path("detection_pt.onnx")

DEFAULT_LABEL_NAMES = {1: "fire", 2: "smoke"}
DEFAULT_NUM_CLASSES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection task example")
    parser.add_argument("--mode", choices=["train", "valid", "test", "export", "tune", "all"], default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="sgd", help="Optimizer name: sgd, adam, adamw, rmsprop")
    parser.add_argument("--checkpoint", default="checkpoints/fasterrcnn_fire_smoke.pt")
    parser.add_argument("--trained-model-dir", default="trained_models")
    parser.add_argument("--trained-model-name", default="detection_trained.pt")
    parser.add_argument("--weights", default="", help="Optional path to pretrained weights. Leave empty to train from scratch.")
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX))
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--gpus-per-trial", type=int, default=1, help="GPUs to allocate per Ray Tune trial when multiple CUDA devices are available.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save one model/optimizer checkpoint per trial for recovery.")
    parser.add_argument("--resume-path", default=None, help="Restore a previous Ray Tune experiment directory.")
    parser.add_argument("--grace-period", type=int, default=1, help="Minimum Tune iterations before ASHA can stop a trial.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of Ray Tune trials to run.")
    return parser.parse_args()


def _train_transforms():
    return Compose([RandomHorizontalFlip(0.5), ToImage(), ToDtype(torch.float32, scale=True)])


def _eval_transforms():
    return Compose([ToImage(), ToDtype(torch.float32, scale=True)])


def build_datasets(root: Path, train_split: str, eval_split: str, test_split: str):
    train_dataset = CocoDetectionDataset(
        images_root=root / train_split,
        annotation_path=root / "annotations" / f"instances_{train_split}.json",
        transforms=_train_transforms(),
    )
    eval_dataset = CocoDetectionDataset(
        images_root=root / eval_split,
        annotation_path=root / "annotations" / f"instances_{eval_split}.json",
        transforms=_eval_transforms(),
    )
    test_dataset = CocoDetectionDataset(
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
        collate_fn=detection_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, eval_dataset, test_dataset, train_loader, eval_loader, test_loader


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    root = Path(args.root)
    train_dataset, eval_dataset, test_dataset, train_loader, eval_loader, test_loader = build_loaders(args)

    task = Detection(device=device, label_names=DEFAULT_LABEL_NAMES)
    if args.mode != "tune":
        task.build_model(num_classes=DEFAULT_NUM_CLASSES)

        if args.weights:
            load_weights_into_model(task.model, args.weights)

    trained_model_path = Path(args.trained_model_dir) / args.trained_model_name
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

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
        task.evaluate_model(
            eval_loader=eval_loader,
            device=device,
            score_threshold=args.score_threshold,
        )

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
            num_classes=DEFAULT_NUM_CLASSES,
        )

    if args.mode == "tune":
        config = {
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([2, 4, 8]),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
        }
        if available_gpus <= 0:
            gpus_per_trial = 0
        elif available_gpus == 1:
            gpus_per_trial = 1
        else:
            gpus_per_trial = min(args.gpus_per_trial, available_gpus)

        task.optimize_parameters(
            config,
            train_dataset=train_dataset,
            device=device,
            epochs=args.epochs,
            eval_dataset=eval_dataset,
            num_classes=DEFAULT_NUM_CLASSES,
            optimizer_name=args.optimizer,
            initial_weights_path=args.weights or None,
            save_checkpoints=args.save_checkpoints,
            resume_path=args.resume_path,
            grace_period=args.grace_period,
            cpus_per_trial=2,
            gpus_per_trial=gpus_per_trial,
            max_num_epochs=args.epochs,
            num_trials=args.num_samples,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, Resize, ToDtype, ToPureTensor

from datasets import ImageBinaryClassificationDataset, ImageClassificationDataset
from examples.common import device_from_arg, load_weights_into_model
from vision_tasks import Classification


DEFAULT_ROOT = Path("data/classification/dataset")
DEFAULT_WEIGHTS = Path("")
DEFAULT_ONNX = Path("classification_pt.onnx")
DEFAULT_INPUT_SIZE = (224, 224)
DEFAULT_LABEL_NAMES = {}
DEFAULT_CRITERION = "cross_entropy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image classification task example")
    parser.add_argument("--mode", choices=["train", "valid", "test", "export", "all"], default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="sgd", help="Optimizer name: sgd, adam, adamw, rmsprop")
    parser.add_argument("--checkpoint", default="checkpoints/classification.pt")
    parser.add_argument("--trained-model-dir", default="trained_models")
    parser.add_argument("--trained-model-name", default="classification_trained.pt")
    parser.add_argument("--weights", default="", help="Optional path to pretrained weights. Leave empty to train from scratch.")
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX))
    parser.add_argument("--output-dir", default="predictions_classification")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=0, help="Set 0 to infer classes from train split folders.")
    parser.add_argument("--dataset-type", choices=["multi", "binary"], default="multi")
    parser.add_argument("--criterion", choices=["cross_entropy", "bce", "bce_logits"], default=DEFAULT_CRITERION)
    return parser.parse_args()


def _train_transforms(image_size: int):
    return Compose([
        RandomHorizontalFlip(0.5),
        Resize((image_size, image_size)),
        ToDtype(torch.float32, scale=True),
        ToPureTensor(),
    ])


def _eval_transforms(image_size: int):
    return Compose([
        Resize((image_size, image_size)),
        ToDtype(torch.float32, scale=True),
        ToPureTensor(),
    ])


def build_datasets(root: Path, train_split: str, eval_split: str, test_split: str, image_size: int, dataset_type: str):
    dataset_cls = ImageBinaryClassificationDataset if dataset_type == "binary" else ImageClassificationDataset
    train_dataset = dataset_cls(
        root_dir=root,
        split=train_split,
        transforms=_train_transforms(image_size),
    )
    eval_dataset = dataset_cls(
        root_dir=root,
        split=eval_split,
        transforms=_eval_transforms(image_size),
        classes=train_dataset.classes,
    )
    test_dataset = dataset_cls(
        root_dir=root,
        split=test_split,
        transforms=_eval_transforms(image_size),
        classes=train_dataset.classes,
    )
    return train_dataset, eval_dataset, test_dataset


def build_loaders(args: argparse.Namespace):
    root = Path(args.root)
    train_dataset, eval_dataset, test_dataset = build_datasets(
        root,
        args.train_split,
        args.eval_split,
        args.test_split,
        args.input_size,
        args.dataset_type,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = args.num_classes if args.num_classes > 0 else len(train_dataset.classes)
    label_names = {idx: class_name for idx, class_name in enumerate(train_dataset.classes)}
    return train_loader, eval_loader, test_loader, num_classes, label_names


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    train_loader, eval_loader, test_loader, num_classes, label_names = build_loaders(args)
    task = Classification(device=device, label_names=label_names)
    output_dim = 1 if args.criterion in {"bce", "bce_logits"} else num_classes
    task.build_model(num_classes=num_classes, output_dim=output_dim)

    if args.weights:
        load_weights_into_model(task.model, args.weights)

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
            num_classes=num_classes,
            criterion=args.criterion,
            output_dim=output_dim,
            optimizer_name=args.optimizer
            pretrained_backbone=False,
        )

    if args.mode in {"valid", "all"}:
        task.evaluate_model(
            eval_loader=eval_loader,
            device=device,
            label_names=label_names,
            criterion=args.criterion,
        )

    if args.mode in {"test", "all"}:
        task.test_model(
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            label_names=label_names,
            max_images=args.max_images,
        )

    if args.mode in {"export", "all"}:
        task.export_onnx(
            onnx_path=args.onnx_path,
            input_size=(args.input_size, args.input_size),
            state_dict_path=args.weights if args.weights else None,
            num_classes=num_classes,
        )


if __name__ == "__main__":
    main()

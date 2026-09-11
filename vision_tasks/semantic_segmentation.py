from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from ray import tune
from ray.air import CheckpointConfig
from ray.tune import RunConfig, schedulers
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, Resize, ToDtype, ToPureTensor
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

from .base import PytorchVisionLab


NUM_CLASSES = 12
BACKGROUND_INDEX = 0
INPUT_SIZE = (360, 480)
DEFAULT_LABEL_NAMES = {idx: f"class_{idx}" for idx in range(NUM_CLASSES)}


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.max() > 1.0:
        images = images / 255.0
    return images


def compute_segmentation_losses(outputs, masks, criterion, aux_weight=0.4):
    out_loss = criterion(outputs["out"], masks)
    loss_components = {"out_loss": out_loss}
    total_loss = out_loss
    if "aux" in outputs:
        aux_loss = criterion(outputs["aux"], masks)
        loss_components["aux_loss"] = aux_loss
        total_loss = total_loss + aux_weight * aux_loss
    return total_loss, loss_components


def update_confusion_matrix(confusion_matrix, preds, targets, num_classes):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    indices = num_classes * targets + preds
    confusion_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def compute_iou_from_confusion(confusion_matrix):
    confusion_matrix = confusion_matrix.to(torch.float32)
    intersection = torch.diag(confusion_matrix)
    ground_truth = confusion_matrix.sum(dim=1)
    predicted = confusion_matrix.sum(dim=0)
    union = ground_truth + predicted - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))


def mask_to_bool_stack(mask: torch.Tensor, ignore_index: int = 0) -> tuple[torch.Tensor, list[int]]:
    class_ids = [int(label) for label in mask.unique().tolist() if int(label) != ignore_index]
    if not class_ids:
        return mask.unsqueeze(0).bool(), []

    bool_masks = torch.stack([(mask == class_id) for class_id in class_ids], dim=0)
    return bool_masks, class_ids


def model_output_to_bool_masks(logits: torch.Tensor, background_index: int = 0) -> tuple[torch.Tensor, list[int]]:
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(logits.shape)}")
        logits = logits[0]

    if logits.ndim != 3:
        raise ValueError(f"Expected shape (C, H, W), got {tuple(logits.shape)}")

    pred = logits.argmax(dim=0)
    class_ids = [int(i) for i in pred.unique().tolist() if int(i) != background_index]

    if not class_ids:
        return pred.unsqueeze(0).bool(), []

    bool_masks = torch.stack([(pred == class_id) for class_id in class_ids], dim=0)
    return bool_masks, class_ids


def load_image_tensor(image_path: str | Path, transforms: Compose | None = None) -> torch.Tensor:
    image = decode_image(str(image_path))
    if transforms is not None:
        image = transforms(image)
    return image


def load_mask_tensor(mask_path: str | Path) -> torch.Tensor:
    with Image.open(mask_path) as mask:
        return F.pil_to_tensor(mask).squeeze(0).to(torch.int64)


def resize_label_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=size,
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).to(torch.int64)


def save_colorized_label_map(label_map: torch.Tensor, save_path: str | Path) -> None:
    palette = torch.tensor(
        [
            [240, 240, 240],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [128, 128, 0],
            [0, 128, 128],
        ],
        dtype=torch.uint8,
    )
    label_map = label_map.detach().cpu().to(torch.long)
    label_map = label_map.clamp(min=0, max=palette.shape[0] - 1)
    color_image = palette[label_map].permute(2, 0, 1).contiguous()
    to_pil_image(color_image).save(save_path)


def visualize_mask_overlay(image: torch.Tensor, masks: torch.Tensor, class_ids: list[int], save_path: str | Path) -> None:
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)

    if masks.numel() == 0 or len(class_ids) == 0:
        to_pil_image(image).save(save_path)
        return

    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "orange",
        "purple",
        "pink",
        "brown",
        "lime",
    ]
    mask_colors = [colors[(class_id - 1) % len(colors)] for class_id in class_ids]
    drawn = draw_segmentation_masks(image, masks.cpu(), alpha=0.6, colors=mask_colors)
    to_pil_image(drawn).save(save_path)


class _SemanticOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model(images)["out"]


class SemanticSegmentation(PytorchVisionLab):
    def __init__(
        self,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
        label_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(device=device, model=model, label_names=label_names or DEFAULT_LABEL_NAMES)
        self.num_classes = NUM_CLASSES
        self.background_index = BACKGROUND_INDEX
        self.criterion: torch.nn.Module | None = None

    def build_model(self, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = False):
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.num_classes = num_classes
        self.model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            aux_loss=True,
        )
        return self.model

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        del optimizer, epochs
        return None

    def train_one_epoch(self, train_loader, optimizer, epoch, criterion=None, device: str | torch.device | None = None):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        criterion = criterion or self.criterion
        if criterion is None:
            raise RuntimeError("Semantic segmentation criterion has not been configured.")

        model.train()
        running_loss = 0.0
        running_loss_components = {}

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = normalize_images(images.to(device))
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            loss, loss_components = compute_segmentation_losses(outputs, masks, criterion)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered at epoch {epoch}, step {step}: {loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            for name, value in loss_components.items():
                running_loss_components[name] = running_loss_components.get(name, 0.0) + float(value.detach().cpu())

            component_log = self.format_loss_components(
                {name: float(value.detach().cpu()) for name, value in loss_components.items()}
            )
            print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | {component_log}")

        num_batches = max(len(train_loader), 1)
        avg_loss = running_loss / num_batches
        avg_loss_components = {name: value / num_batches for name, value in sorted(running_loss_components.items())}
        return avg_loss, avg_loss_components

    def train_model(
        self,
        train_loader,
        device: str | torch.device | None = None,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        trained_model_path: str | Path | None = None,
        eval_loader=None,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)
        if self.model is None:
            self.build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
        self.ensure_model()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = self.configure_optimizer(
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.scheduler = self.build_scheduler(self.optimizer, epochs)

        self.history = []
        self.eval_history = []
        final_loss_components = {}
        for epoch in range(1, epochs + 1):
            avg_loss, avg_loss_components = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                criterion=self.criterion,
            )
            self.history.append(avg_loss)
            final_loss_components = avg_loss_components
            print(f"Epoch {epoch} finished | {self.format_loss_components(avg_loss_components)}")

            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate_model(eval_loader=eval_loader)
                self.eval_history.append(eval_metrics)

            if checkpoint_path is not None:
                payload = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "avg_loss": avg_loss,
                    "loss_components": avg_loss_components,
                    "history": list(self.history),
                    "eval_metrics": eval_metrics,
                    "eval_history": list(self.eval_history),
                }
                self.save_checkpoint(checkpoint_path, payload)

            print("=================")

        if trained_model_path is not None:
            self.trained_model_path = self.save_model_state_dict(trained_model_path, self.ensure_model())

        return self.model, self.history, self.eval_history, final_loss_components

    def evaluate_model(
        self,
        eval_loader,
        device: str | torch.device | None = None,
        num_classes: int | None = None,
        background_index: int | None = None,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        num_classes = num_classes or self.num_classes
        background_index = self.background_index if background_index is None else background_index

        was_training = model.training
        model.eval()

        criterion = self.criterion or torch.nn.CrossEntropyLoss(ignore_index=255)
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_pixels = 0
        total_loss_components = {}
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        with torch.inference_mode():
            for images, masks in eval_loader:
                images = normalize_images(images.to(device))
                masks = masks.to(device, dtype=torch.long)

                outputs = model(images)
                logits = outputs["out"]
                loss, loss_components = compute_segmentation_losses(outputs, masks, criterion)

                batch_size = images.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_size
                total_samples += batch_size
                for name, value in loss_components.items():
                    total_loss_components[name] = total_loss_components.get(name, 0.0) + float(value.detach().cpu()) * batch_size

                preds = logits.argmax(dim=1)
                total_correct += int((preds == masks).sum().item())
                total_pixels += int(masks.numel())
                update_confusion_matrix(confusion_matrix, preds.cpu(), masks.cpu(), num_classes)

        if was_training:
            model.train()
        else:
            model.eval()

        avg_loss = total_loss / max(total_samples, 1)
        avg_loss_components = {name: value / max(total_samples, 1) for name, value in sorted(total_loss_components.items())}
        pixel_accuracy = total_correct / max(total_pixels, 1)
        iou_per_class = compute_iou_from_confusion(confusion_matrix)
        valid_class_mask = torch.ones(num_classes, dtype=torch.bool)
        if 0 <= background_index < num_classes:
            valid_class_mask[background_index] = False

        foreground_iou = iou_per_class[valid_class_mask]
        mean_iou = (
            float(foreground_iou.mean().item())
            if foreground_iou.numel() > 0
            else float(iou_per_class.mean().item())
        )
        class_iou = {class_id: float(iou_per_class[class_id].item()) for class_id in range(num_classes)}

        metrics = {
            "loss": avg_loss,
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "class_iou": class_iou,
            "loss_components": avg_loss_components,
        }

        class_iou_text = ", ".join(f"c{class_id}={value:.4f}" for class_id, value in class_iou.items())
        print(
            f"Test | {self.format_loss_components(metrics['loss_components'])} | pixel_acc={metrics['pixel_accuracy']:.4f} | "
            f"mean_iou={metrics['mean_iou']:.4f} | {class_iou_text}"
        )
        return metrics

    def predict_image(
        self,
        image_path: str | Path,
        transforms: Compose,
        device: str | torch.device | None = None,
        background_index: int | None = None,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        background_index = self.background_index if background_index is None else background_index
        image = load_image_tensor(image_path, transforms=transforms)

        with torch.inference_mode():
            logits = model(image.to(device).unsqueeze(0))["out"]
        pred_mask, pred_class_ids = model_output_to_bool_masks(logits=logits, background_index=background_index)
        return image, logits, pred_mask, pred_class_ids

    def visualize_prediction(
        self,
        image: torch.Tensor,
        output,
        save_path: str | Path,
        mask_path: str | Path | None = None,
        background_index: int | None = None,
    ) -> None:
        background_index = self.background_index if background_index is None else background_index
        if isinstance(output, dict):
            logits = output["out"]
        else:
            logits = output

        if logits.ndim == 4:
            logits = logits[0]

        pred_mask, pred_class_ids = model_output_to_bool_masks(logits=logits, background_index=background_index)
        visualize_mask_overlay(image=image, masks=pred_mask, class_ids=pred_class_ids, save_path=save_path)

        pred_label_map = logits.argmax(dim=0)
        pred_map_path = str(Path(save_path).with_name(Path(save_path).stem + "_labels.png"))
        save_colorized_label_map(pred_label_map, pred_map_path)

        if mask_path is not None:
            gt_mask = load_mask_tensor(mask_path)
            gt_mask = resize_label_mask(gt_mask, size=pred_label_map.shape[-2:])
            gt_bool_mask, gt_class_ids = mask_to_bool_stack(mask=gt_mask)
            gt_save_path = str(Path(save_path).with_name(Path(save_path).stem + "_gt.png"))
            visualize_mask_overlay(image=image, masks=gt_bool_mask, class_ids=gt_class_ids, save_path=gt_save_path)

    def test_model(
        self,
        test_loader,
        device: str | torch.device | None = None,
        output_dir: str | Path = "predictions_semantic",
        background_index: int | None = None,
        max_images: int = 12,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        background_index = self.background_index if background_index is None else background_index
        model.to(device)
        model.eval()

        output_dir = self.ensure_dir(output_dir)
        saved_images = 0

        with torch.inference_mode():
            for step, (images, masks) in enumerate(test_loader, start=1):
                images = normalize_images(images.to(device))
                outputs = model(images)
                logits = outputs["out"]

                for image_idx in range(images.shape[0]):
                    image = images[image_idx].detach().cpu()
                    output = logits[image_idx].detach().cpu()
                    save_path = output_dir / f"pred_{step:04d}_{image_idx:02d}.png"
                    self.visualize_prediction(
                        image=image,
                        output=output,
                        save_path=save_path,
                        mask_path=None,
                        background_index=background_index,
                    )
                    if masks is not None:
                        gt_mask = masks[image_idx].detach().cpu()
                        gt_bool_mask, gt_class_ids = mask_to_bool_stack(mask=gt_mask)
                        gt_save_path = output_dir / f"pred_{step:04d}_{image_idx:02d}_gt.png"
                        visualize_mask_overlay(
                            image=image,
                            masks=gt_bool_mask,
                            class_ids=gt_class_ids,
                            save_path=gt_save_path,
                        )

                    print(f"saved {save_path}")
                    saved_images += 1
                    if saved_images >= max_images:
                        return

    def export_onnx(
        self,
        onnx_path: str | Path = "semantic_segmentation_pt.onnx",
        input_size: tuple[int, int] = (480, 480),
        state_dict_path: str | Path | None = None,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = False,
    ) -> Path:
        if state_dict_path is not None:
            model = self.build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
            state_dict = torch.load(state_dict_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.model = model
        else:
            model = self.ensure_model() if self.model is not None else self.build_model(
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
            )

        model = model.to(self.device).eval()
        wrapper = _SemanticOnnxWrapper(model).eval()
        dummy_input = torch.rand((1, 3, input_size[0], input_size[1]), device=self.device)
        onnx_path = Path(onnx_path)

        onnx_program = torch.onnx.export(
            model=wrapper,
            args=dummy_input,
            input_names=["input"],
            output_names=["output"],
            verbose=True,
        )
        if hasattr(onnx_program, "save"):
            onnx_program.save(str(onnx_path))
        else:
            torch.onnx.export(
                model=wrapper,
                args=dummy_input,
                f=str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                verbose=True,
            )
        return onnx_path

    def trial(
        self,
        config,
        train_dataset=None,
        device: str | torch.device | None = None,
        epochs: int = 10,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        eval_dataset=None,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = False,
        initial_weights_path: str | Path | None = None,
        save_checkpoints: bool = False,
    ):
        if train_dataset is None or eval_dataset is None:
            raise ValueError("Ray Tune requires both train_dataset and eval_dataset for semantic segmentation.")

        if device is not None:
            self.device = torch.device(device)

        self.history = []
        self.eval_history = []

        if self.model is None:
            self.build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
        if initial_weights_path is not None:
            self.load_model_state_dict(self.model, initial_weights_path)
        self.ensure_model()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = self.configure_optimizer(
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.scheduler = self.build_scheduler(self.optimizer, epochs)

        checkpoint = tune.get_checkpoint()
        if checkpoint is not None:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
                checkpoint_state = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint_state, dict):
                    model_state = checkpoint_state.get("model_state_dict")
                    optimizer_state = checkpoint_state.get("optimizer_state_dict")
                else:
                    model_state, optimizer_state = checkpoint_state
                if model_state is not None:
                    self.model.load_state_dict(model_state)
                if optimizer_state is not None:
                    self.optimizer.load_state_dict(optimizer_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
        )

        for epoch in range(1, epochs + 1):
            avg_loss, avg_loss_components = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                criterion=self.criterion,
            )
            self.history.append(avg_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            eval_metrics = self.evaluate_model(eval_loader=eval_loader)
            self.eval_history.append(eval_metrics)

            report_metrics = {
                "epoch": epoch,
                "train_loss": avg_loss,
            }
            report_metrics.update(self.flatten_numeric_metrics({"train_loss_components": avg_loss_components}))
            report_metrics.update(self.flatten_numeric_metrics(eval_metrics))

            if save_checkpoints:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    checkpoint_path = Path(temp_checkpoint_dir) / "checkpoint.pt"
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                        },
                        checkpoint_path,
                    )
                    tune.report(report_metrics, checkpoint=tune.Checkpoint.from_directory(temp_checkpoint_dir))
            else:
                tune.report(report_metrics)

    def optimize_parameters(
        self,
        config,
        train_dataset=None,
        device: str | torch.device | None = None,
        epochs: int = 10,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        eval_dataset=None,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = False,
        initial_weights_path: str | Path | None = None,
        save_checkpoints: bool = False,
        resume_path: str | Path | None = None,
        cpus_per_trial: int = 2,
        gpus_per_trial: int = 1,
        max_num_epochs: int | None = None,
        grace_period: int = 1,
        num_trials: int = 10,
    ):
        if eval_dataset is None:
            raise ValueError("Ray Tune requires eval_dataset for semantic segmentation.")

        if resume_path is not None and not save_checkpoints:
            raise ValueError("resume_path requires save_checkpoints=True so resumed trials can continue writing checkpoints.")

        if max_num_epochs is None:
            max_num_epochs = epochs

        tune_scheduler = schedulers.ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=2,
        )

        trainable = tune.with_resources(
                tune.with_parameters(
                    self.trial,
                    train_dataset=train_dataset,
                    device=device,
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    eval_dataset=eval_dataset,
                    num_classes=num_classes,
                    pretrained_backbone=pretrained_backbone,
                    initial_weights_path=initial_weights_path,
                    save_checkpoints=save_checkpoints,
                    epochs=epochs,
                ),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            )
        tune_config = tune.TuneConfig(
                metric="mean_iou",
                mode="max",
                scheduler=tune_scheduler,
                num_samples=num_trials,
            )
        run_config = RunConfig(
            storage_path=str(Path.cwd() / "ray_tune") if save_checkpoints else "/tmp/ray_tune",
            verbose=1,
            log_to_file=False,
            checkpoint_config=CheckpointConfig(num_to_keep=1) if save_checkpoints else None,
        )

        if resume_path is not None:
            if not tune.Tuner.can_restore(resume_path):
                raise ValueError(f"Ray Tune cannot restore from '{resume_path}'. Expected an experiment directory containing experiment_state*.json.")
            tuner = tune.Tuner.restore(
                str(resume_path),
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
                param_space=config,
            )
        else:
            tuner = tune.Tuner(
                trainable,
                tune_config=tune_config,
                param_space=config,
                run_config=run_config,
            )

        results = tuner.fit()
        best_result = results.get_best_result("mean_iou", "max")

        print(f"Best trial config: {best_result.config}")
        print(f"Best trial mean_iou: {best_result.metrics.get('mean_iou')}")
        print(f"Best trial pixel_accuracy: {best_result.metrics.get('pixel_accuracy')}")
        print(f"Best trial loss: {best_result.metrics.get('loss')}")
        return best_result


def _task_from_model(model, device=None):
    return SemanticSegmentation(device=device, model=model)


def build_model(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = False):
    return SemanticSegmentation().build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)


def train_model(
    model,
    train_loader,
    device,
    epochs=10,
    lr=0.001,
    weight_decay=0.0005,
    checkpoint_path=None,
    trained_model_path=None,
    eval_loader=None,
    optimizer_name="sgd",
    optimizer_kwargs=None,
):
    return _task_from_model(model, device=device).train_model(
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
        trained_model_path=trained_model_path,
        eval_loader=eval_loader,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
    )


def evaluate_model(
    model,
    test_loader,
    device,
    num_classes=NUM_CLASSES,
    background_index=BACKGROUND_INDEX,
):
    return _task_from_model(model, device=device).evaluate_model(
        eval_loader=test_loader,
        device=device,
        num_classes=num_classes,
        background_index=background_index,
    )


def visualize_prediction(image, output, save_path, mask_path=None, background_index=BACKGROUND_INDEX):
    return _task_from_model(None).visualize_prediction(
        image=image,
        output=output,
        save_path=save_path,
        mask_path=mask_path,
        background_index=background_index,
    )


def test_model(
    model,
    test_loader,
    device,
    output_dir="predictions_semantic",
    background_index=BACKGROUND_INDEX,
    max_images=12,
):
    return _task_from_model(model, device=device).test_model(
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        background_index=background_index,
        max_images=max_images,
    )


def export_onnx(
    model=None,
    onnx_path="semantic_segmentation_pt.onnx",
    input_size=(480, 480),
    state_dict_path=None,
    num_classes=NUM_CLASSES,
    pretrained_backbone=False,
):
    return _task_from_model(model).export_onnx(
        onnx_path=onnx_path,
        input_size=input_size,
        state_dict_path=state_dict_path,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )


def predict_and_visualize(
    model,
    image_path: str | Path,
    device: str,
    transforms: Compose,
    save_path: str = "test_segmentation_pred.png",
    mask_path: str | Path | None = None,
):
    task = _task_from_model(model, device=device)
    image, logits, pred_mask, pred_class_ids = task.predict_image(
        image_path=image_path,
        transforms=transforms,
        device=device,
    )
    task.visualize_prediction(
        image=image,
        output=logits.squeeze(0),
        save_path=save_path,
        mask_path=mask_path,
    )
    return logits, pred_class_ids, []


def load_segmentation_model(model_path: str | Path, device: str, num_classes: int = NUM_CLASSES):
    model_weights = torch.load(model_path, map_location=device)
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes, aux_loss=True)
    model.load_state_dict(model_weights)
    return model.to(device).eval()


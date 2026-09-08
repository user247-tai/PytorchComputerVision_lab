from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.air import CheckpointConfig
from ray.tune import RunConfig, schedulers
from torch.utils.data import DataLoader
from helper.coco_eval import CocoEvaluator
from helper.coco_utils import get_coco_api_from_dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from datasets import instance_collate_fn

from .base import PytorchVisionLab


DEFAULT_LABEL_NAMES = {1: "class_1"}


def _mask_iou(pred_mask: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    pred_mask = pred_mask.to(torch.bool).reshape(-1)
    if gt_masks.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    gt_masks = gt_masks.to(torch.bool).reshape(gt_masks.shape[0], -1)
    intersection = torch.logical_and(gt_masks, pred_mask.unsqueeze(0)).sum(dim=1).to(torch.float32)
    union = torch.logical_or(gt_masks, pred_mask.unsqueeze(0)).sum(dim=1).to(torch.float32)
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def _evaluate_class_at_iou(predictions, ground_truths, class_id, iou_threshold, score_threshold=None):
    gt_masks_by_image = []
    matched_flags_by_image = []
    total_gt = 0

    for gt in ground_truths:
        gt_mask = gt["labels"] == class_id
        gt_masks = gt["masks"][gt_mask].to(torch.bool)
        gt_masks_by_image.append(gt_masks)
        matched_flags_by_image.append(torch.zeros(len(gt_masks), dtype=torch.bool))
        total_gt += len(gt_masks)

    pred_records = []
    for image_idx, pred in enumerate(predictions):
        pred_mask = pred["labels"] == class_id
        pred_masks = pred["masks"][pred_mask].to(torch.bool)
        pred_scores = pred["scores"][pred_mask].float()

        if score_threshold is not None:
            keep = pred_scores >= score_threshold
            pred_masks = pred_masks[keep]
            pred_scores = pred_scores[keep]

        for mask, score in zip(pred_masks, pred_scores):
            pred_records.append((image_idx, float(score), mask))

    pred_records.sort(key=lambda item: item[1], reverse=True)

    tp = []
    fp = []
    matched_ious = []

    for image_idx, _, pred_mask in pred_records:
        gt_masks = gt_masks_by_image[image_idx]
        if gt_masks.numel() == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = _mask_iou(pred_mask, gt_masks)
        best_iou, best_idx = ious.max(dim=0)

        if float(best_iou) >= iou_threshold and not matched_flags_by_image[image_idx][best_idx]:
            matched_flags_by_image[image_idx][best_idx] = True
            tp.append(1)
            fp.append(0)
            matched_ious.append(float(best_iou))
        else:
            tp.append(0)
            fp.append(1)

    if total_gt == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "mean_iou": 0.0,
            "num_gt": 0,
        }

    if tp:
        tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
        fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
        recall = tp_cum / total_gt
        ap = PytorchVisionLab.compute_average_precision(recall, precision)
        precision_value = float(precision[-1])
        recall_value = float(recall[-1])
        tp_total = int(tp_cum[-1].item())
        fp_total = int(fp_cum[-1].item())
    else:
        ap = 0.0
        precision_value = 0.0
        recall_value = 0.0
        tp_total = 0
        fp_total = 0

    fn_total = total_gt - tp_total
    mean_iou = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

    return {
        "ap": ap,
        "precision": precision_value,
        "recall": recall_value,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "mean_iou": mean_iou,
        "num_gt": total_gt,
    }


class _InstanceOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        output = self.model([images[0]])[0]
        masks = output["masks"]
        if masks.ndim == 4:
            masks = masks[:, 0]
        return output["boxes"], output["labels"], output["scores"], masks


class InstanceSegmentation(PytorchVisionLab):
    def __init__(
        self,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
        label_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(device=device, model=model, label_names=label_names or DEFAULT_LABEL_NAMES)
        self.freeze_backbone = True

    def build_model(self, num_classes: int = 2, freeze_backbone: bool = True, pretrained_backbone: bool = False):
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = mobilenet_backbone(
            backbone_name="mobilenet_v3_large",
            weights=weights_backbone,
            fpn=True,
            trainable_layers=0 if freeze_backbone else 2,
        )

        anchor_generator = AnchorGenerator(
            sizes=((16, 32), (64, 128), (256, 512)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3,
        )
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "pool"], output_size=7, sampling_ratio=2)
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "pool"], output_size=14, sampling_ratio=2)

        self.freeze_backbone = freeze_backbone
        self.model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            min_size=300,
            max_size=500,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool,
            box_nms_thresh=0.1,
        )
        if freeze_backbone:
            self.freeze_model_backbone(self.model)
        return self.model

    @staticmethod
    def freeze_model_backbone(model) -> None:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        del epochs
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        epoch,
        device: str | torch.device | None = None,
        warmup_scheduler=None,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        model.train()
        running_loss = 0.0
        running_loss_components = {}

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [self.move_to_device(target, device) for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                raise RuntimeError(
                    f"Non-finite loss encountered at epoch {epoch}, step {step}: {loss_dict}"
                )

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            running_loss += float(losses.detach().cpu())
            for name, value in loss_dict.items():
                running_loss_components[name] = running_loss_components.get(name, 0.0) + float(value.detach().cpu())

            component_log = self.format_loss_components(
                {name: float(value.detach().cpu()) for name, value in sorted(loss_dict.items())}
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
        lr: float = 0.005,
        weight_decay: float = 0.0005,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        trained_model_path: str | Path | None = None,
        eval_loader=None,
        pretrained_backbone: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)
        if self.model is None:
            self.build_model(pretrained_backbone=pretrained_backbone)
        self.ensure_model()
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
            warmup_scheduler = None
            if epoch == 1:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(train_loader) - 1)
                if warmup_iters > 0:
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=warmup_factor,
                        total_iters=warmup_iters,
                    )

            avg_loss, avg_loss_components = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                warmup_scheduler=warmup_scheduler,
            )
            self.scheduler.step()
            self.history.append(avg_loss)
            final_loss_components = avg_loss_components

            component_log = self.format_loss_components(avg_loss_components)
            print(f"Epoch {epoch} finished | {component_log}")

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
                    "history": list(self.history),
                    "eval_history": list(self.eval_history),
                    "eval_metrics": eval_metrics,
                    "loss_components": avg_loss_components,
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
        score_threshold: float = 0.5,
        label_names: dict[int, str] | None = None,
    ):
        del score_threshold, label_names
        model = self.ensure_model()
        device = torch.device(device or self.device)
        was_training = model.training
        model.eval()

        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")

        coco = get_coco_api_from_dataset(eval_loader.dataset)
        coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])

        total_loss = 0.0
        total_images = 0
        running_loss_components = {}

        for images, targets in eval_loader:
            images_device = [image.to(device) for image in images]
            targets_device = [self.move_to_device(target, device) for target in targets]

            model.train()
            self.freeze_batchnorm_layers(model)
            loss_dict = model(images_device, targets_device)
            batch_loss = sum(loss for loss in loss_dict.values())

            total_loss += float(batch_loss.detach().cpu()) * len(images_device)
            total_images += len(images_device)
            for name, value in loss_dict.items():
                running_loss_components[name] = running_loss_components.get(name, 0.0) + float(value.detach().cpu()) * len(
                    images_device
                )

            model.eval()
            outputs = model(images_device)
            outputs = [{k: v.to(cpu_device) for k, v in output.items()} for output in outputs]

            predictions = {}
            for target, output in zip(targets, outputs):
                image_id = target["image_id"]
                if torch.is_tensor(image_id):
                    image_id = int(image_id.item())
                else:
                    image_id = int(image_id)
                predictions[image_id] = output
            coco_evaluator.update(predictions)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)

        if was_training:
            model.train()
        else:
            model.eval()

        avg_loss = total_loss / max(total_images, 1)
        avg_loss_components = {name: value / max(total_images, 1) for name, value in sorted(running_loss_components.items())}
        bbox_stats = coco_evaluator.coco_eval["bbox"].stats
        segm_stats = coco_evaluator.coco_eval["segm"].stats
        metrics = {
            "loss": avg_loss,
            "loss_components": avg_loss_components,
            "bbox_ap": float(bbox_stats[0]),
            "bbox_ap50": float(bbox_stats[1]),
            "bbox_ap75": float(bbox_stats[2]),
            "segm_ap": float(segm_stats[0]),
            "segm_ap50": float(segm_stats[1]),
            "segm_ap75": float(segm_stats[2]),
            "num_images": total_images,
        }

        component_log = self.format_loss_components(avg_loss_components)
        print(f"Test | {component_log} | bbox_ap={metrics['bbox_ap']:.4f} | segm_ap={metrics['segm_ap']:.4f}")
        return metrics

    def visualize_prediction(
        self,
        image: torch.Tensor,
        output: dict,
        save_path: str | Path,
        score_threshold: float = 0.5,
        label_names: dict[int, str] | None = None,
    ) -> None:
        image = self.image_to_uint8(image)
        label_names = label_names or self.label_names or DEFAULT_LABEL_NAMES

        keep = output["scores"].detach().cpu() > score_threshold
        boxes = output["boxes"].detach().cpu()[keep]
        labels = output["labels"].detach().cpu()[keep]
        scores = output["scores"].detach().cpu()[keep]
        masks = output["masks"].detach().cpu()[keep, 0] > 0.5

        if boxes.numel() == 0:
            to_pil_image(image).save(save_path)
            return

        label_texts = [
            f"{label_names.get(int(label_id), str(int(label_id)))}: {float(score):.3f}"
            for label_id, score in zip(labels, scores)
        ]
        drawn = draw_segmentation_masks(image, masks, alpha=0.5, colors="red")
        drawn = draw_bounding_boxes(drawn, boxes.round().to(torch.int64), labels=label_texts, colors="green", width=3)
        to_pil_image(drawn).save(save_path)

    def test_model(
        self,
        test_loader,
        device: str | torch.device | None = None,
        output_dir: str | Path = "predictions_instance",
        score_threshold: float = 0.5,
        label_names: dict[int, str] | None = None,
        max_images: int = 12,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        model.to(device)
        model.eval()

        label_names = label_names or self.label_names or DEFAULT_LABEL_NAMES
        output_dir = self.ensure_dir(output_dir)

        saved_images = 0
        with torch.inference_mode():
            for step, (images, targets) in enumerate(test_loader, start=1):
                del targets
                images = [image.to(device) for image in images]
                outputs = model(images)

                for image_idx, (image, output) in enumerate(zip(images, outputs)):
                    out_path = output_dir / f"pred_{step:04d}_{image_idx:02d}.png"
                    self.visualize_prediction(
                        image=image,
                        output=output,
                        save_path=out_path,
                        score_threshold=score_threshold,
                        label_names=label_names,
                    )
                    print(f"saved {out_path}")
                    saved_images += 1

                    if saved_images >= max_images:
                        return

    def export_onnx(
        self,
        onnx_path: str | Path = "instance_pt.onnx",
        input_size: tuple[int, int] = (480, 480),
        state_dict_path: str | Path | None = None,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = False,
    ) -> Path:
        if state_dict_path is not None:
            model = self.build_model(
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
                pretrained_backbone=pretrained_backbone,
            )
            state_dict = torch.load(state_dict_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.model = model
        else:
            model = self.ensure_model() if self.model is not None else self.build_model(
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
                pretrained_backbone=pretrained_backbone,
            )

        model = model.to(self.device).eval()
        wrapper = _InstanceOnnxWrapper(model)
        dummy_input = torch.rand((1, 3, input_size[0], input_size[1]), device=self.device)
        onnx_path = Path(onnx_path)

        torch.onnx.export(
            model=wrapper,
            args=dummy_input,
            input_names=["input"],
            output_names=["boxes", "labels", "scores", "masks"],
            verbose=True,
            dynamo=False,
            opset_version=17,
            f=str(onnx_path),
            dynamic_axes={
                "boxes": {0: "num_detections"},
                "labels": {0: "num_detections"},
                "scores": {0: "num_detections"},
                "masks": {0: "num_detections"},
            },
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
        num_classes: int = 2,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = False,
        initial_weights_path: str | Path | None = None,
        save_checkpoints: bool = False,
    ):
        if train_dataset is None or eval_dataset is None:
            raise ValueError("Ray Tune requires both train_dataset and eval_dataset for instance segmentation.")

        if device is not None:
            self.device = torch.device(device)

        self.history = []
        self.eval_history = []

        if self.model is None:
            self.build_model(
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
                pretrained_backbone=pretrained_backbone,
            )
        if initial_weights_path is not None:
            self.load_model_state_dict(self.model, initial_weights_path)
        self.ensure_model()

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

        for epoch in range(1, epochs + 1):
            warmup_scheduler = None
            if epoch == 1:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(train_loader) - 1)
                if warmup_iters > 0:
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=warmup_factor,
                        total_iters=warmup_iters,
                    )

            avg_loss, avg_loss_components = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                warmup_scheduler=warmup_scheduler,
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
        num_classes: int = 2,
        freeze_backbone: bool = True,
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
            raise ValueError("Ray Tune requires eval_dataset for instance segmentation.")

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
                    freeze_backbone=freeze_backbone,
                    pretrained_backbone=pretrained_backbone,
                    initial_weights_path=initial_weights_path,
                    save_checkpoints=save_checkpoints,
                    epochs=epochs,
                ),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            )
        tune_config = tune.TuneConfig(
                metric="segm_ap",
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
        best_result = results.get_best_result("segm_ap", "max")

        print(f"Best trial config: {best_result.config}")
        print(f"Best trial bbox_ap: {best_result.metrics.get('bbox_ap')}")
        print(f"Best trial segm_ap: {best_result.metrics.get('segm_ap')}")
        print(f"Best trial loss: {best_result.metrics.get('loss')}")
        return best_result


def _task_from_model(model, device=None):
    return InstanceSegmentation(device=device, model=model)


def build_model(num_classes: int = 2, freeze_backbone: bool = True, pretrained_backbone: bool = False):
    return InstanceSegmentation().build_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained_backbone=pretrained_backbone,
    )


def train_model(
    model,
    train_loader,
    device,
    epochs=10,
    lr=0.005,
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
    score_threshold=0.5,
    label_names=None,
):
    return _task_from_model(model, device=device).evaluate_model(
        eval_loader=test_loader,
        device=device,
        score_threshold=score_threshold,
        label_names=label_names,
    )


def visualize_prediction(image, output, save_path, score_threshold=0.5, label_names=None):
    return _task_from_model(None).visualize_prediction(
        image=image,
        output=output,
        save_path=save_path,
        score_threshold=score_threshold,
        label_names=label_names,
    )


def test_model(
    model,
    test_loader,
    device,
    output_dir="predictions_instance",
    score_threshold=0.5,
    label_names=None,
    max_images=12,
):
    return _task_from_model(model, device=device).test_model(
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        score_threshold=score_threshold,
        label_names=label_names,
        max_images=max_images,
    )


def export_onnx(
    model=None,
    onnx_path="instance_pt.onnx",
    input_size=(480, 480),
    state_dict_path=None,
    num_classes=2,
    freeze_backbone=True,
    pretrained_backbone=False,
):
    return _task_from_model(model).export_onnx(
        onnx_path=onnx_path,
        input_size=input_size,
        state_dict_path=state_dict_path,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained_backbone=pretrained_backbone,
    )


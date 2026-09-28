from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from functools import partial
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_pil_image
from ray import tune
from ray.air import CheckpointConfig
from ray.tune import RunConfig, schedulers
import os
import tempfile

from .base import PytorchVisionLab


DEFAULT_LABEL_NAMES: dict[int, str] = {}


def _normalize_logits_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 1:
        return output.unsqueeze(0)
    return output


class Classification(PytorchVisionLab):
    def __init__(
        self,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
        label_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(device=device, model=model, label_names=label_names or DEFAULT_LABEL_NAMES)
        self.num_classes = 0
        self.output_dim = 0
        self.criterion: torch.nn.Module | None = None

    def build_model(self, num_classes: int, pretrained_backbone: bool = False, output_dim: int | None = None):
        weights = ResNet18_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        model = resnet18(weights=weights)
        output_dim = output_dim or num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, output_dim)
        self.model = model
        self.num_classes = num_classes
        self.output_dim = output_dim
        return self.model

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        del optimizer, epochs
        return None

    @staticmethod
    def _is_binary_criterion(criterion: torch.nn.Module | None) -> bool:
        return isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss))

    @staticmethod
    def _normalize_logits_for_output(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            return logits.unsqueeze(0)
        return logits

    @staticmethod
    def _predict_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = Classification._normalize_logits_for_output(logits)
        if logits.shape[-1] == 1:
            return (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        return logits.argmax(dim=1)

    @staticmethod
    def _probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = Classification._normalize_logits_for_output(logits)
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
            return torch.stack([1 - probs, probs], dim=-1)
        return torch.softmax(logits, dim=-1)

    def _resolve_criterion(
        self,
        criterion: torch.nn.Module | str | None,
    ) -> torch.nn.Module:
        if isinstance(criterion, str):
            name = criterion.lower().strip()
            if name == "cross_entropy":
                return nn.CrossEntropyLoss()
            if name == "bce":
                return nn.BCELoss()
            if name == "bce_logits":
                return nn.BCEWithLogitsLoss()
            raise ValueError(
                f"Unsupported classification criterion '{criterion}'. "
                "Supported values: cross_entropy, bce, bce_logits"
            )

        raise ValueError(
            "Classification criterion must be one of: cross_entropy, bce, bce_logits"
        )

    def _prepare_targets_for_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._normalize_logits_for_output(logits)
        if self._is_binary_criterion(criterion):
            logits = logits.squeeze(-1)
            targets = targets.to(device=logits.device, dtype=torch.float32).view_as(logits)
            if isinstance(criterion, nn.BCELoss):
                logits = torch.sigmoid(logits)
            return logits, targets

        targets = targets.to(device=logits.device, dtype=torch.long)
        return logits, targets

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        epoch,
        criterion=None,
        device: str | torch.device | None = None,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        criterion = criterion or self.criterion
        if criterion is None:
            raise RuntimeError("Classification criterion has not been configured.")

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            logits_for_loss, targets_for_loss = self._prepare_targets_for_loss(logits, targets, criterion)
            loss = criterion(logits_for_loss, targets_for_loss)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}, step {step}: {float(loss.detach().cpu())}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = targets.shape[0]
            preds = self._predict_from_logits(logits.detach())
            running_correct += int((preds.cpu() == targets.detach().cpu().long()).sum().item())
            running_samples += batch_size
            running_loss += float(loss.detach().cpu())

            batch_acc = running_correct / max(running_samples, 1)
            print(
                f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                f"loss={float(loss.detach().cpu()):.4f} | acc={batch_acc:.4f}"
            )

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_acc = running_correct / max(running_samples, 1)
        return avg_loss, {"accuracy": avg_acc}

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
        num_classes: int = 2,
        criterion: torch.nn.Module | str | None = "cross_entropy",
        output_dim: int | None = None,
        pretrained_backbone: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)

        resolved_output_dim = output_dim
        if resolved_output_dim is None:
            if isinstance(criterion, str):
                is_binary_criterion = criterion.lower().strip() in {"bce", "bce_logits"}
            else:
                is_binary_criterion = False
            resolved_output_dim = 1 if is_binary_criterion else num_classes

        self.criterion = self._resolve_criterion(criterion)

        if self.model is None or self.num_classes != num_classes or self.output_dim != resolved_output_dim:
            self.build_model(
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
                output_dim=resolved_output_dim,
            )
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
        final_train_metrics: dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            avg_loss, train_metrics = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                criterion=self.criterion,
            )
            self.history.append(avg_loss)
            final_train_metrics = train_metrics

            train_log = self.format_loss_components({"loss": avg_loss, **train_metrics})
            print(f"Epoch {epoch} finished | {train_log}")

            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate_model(eval_loader=eval_loader, criterion=criterion)
                self.eval_history.append(eval_metrics)

            if checkpoint_path is not None:
                payload = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "avg_loss": avg_loss,
                    "history": list(self.history),
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "eval_history": list(self.eval_history),
                }
                self.save_checkpoint(checkpoint_path, payload)

            print("=================")

        if trained_model_path is not None:
            self.trained_model_path = self.save_model_state_dict(trained_model_path, self.ensure_model())

        return self.model, self.history, self.eval_history, final_train_metrics

    def evaluate_model(
        self,
        eval_loader,
        device: str | torch.device | None = None,
        label_names: dict[int, str] | None = None,
        criterion: torch.nn.Module | str | None = "cross_entropy",
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        was_training = model.training
        model.eval()

        label_names = label_names or self.label_names or DEFAULT_LABEL_NAMES
        num_classes = self.num_classes or len(label_names)
        resolved_criterion = self._resolve_criterion(criterion or self.criterion)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        with torch.inference_mode():
            for images, targets in eval_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                logits_for_loss, targets_for_loss = self._prepare_targets_for_loss(logits, targets, resolved_criterion)
                loss = resolved_criterion(logits_for_loss, targets_for_loss)
                preds = self._predict_from_logits(logits.detach())

                batch_size = targets.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_size
                total_correct += int((preds.cpu() == targets.detach().cpu().long()).sum().item())
                total_samples += batch_size

                flat_targets = targets.detach().cpu().long().reshape(-1)
                flat_preds = preds.detach().cpu().long().reshape(-1)
                indices = num_classes * flat_targets + flat_preds
                confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

        if was_training:
            model.train()
        else:
            model.eval()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        class_accuracy = {}
        row_sums = confusion.sum(dim=1)
        for class_id in range(num_classes):
            class_accuracy[class_id] = float(confusion[class_id, class_id].item() / max(int(row_sums[class_id].item()), 1))

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "num_samples": total_samples,
        }

        class_log = ", ".join(
            f"{label_names.get(class_id, str(class_id))}={value:.4f}" for class_id, value in class_accuracy.items()
        )
        print(f"Eval | loss={avg_loss:.4f} | acc={accuracy:.4f} | {class_log}")
        return metrics

    def _prepare_image_for_display(self, image: torch.Tensor) -> Image.Image:
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        return to_pil_image(image.clamp(0, 1) if image.dtype.is_floating_point else image)

    def visualize_prediction(
        self,
        image: torch.Tensor,
        output,
        save_path: str | Path,
        label_names: dict[int, str] | None = None,
        topk: int = 3,
    ) -> None:
        label_names = label_names or self.label_names or DEFAULT_LABEL_NAMES
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output", output))
        else:
            logits = output
        logits = self._normalize_logits_for_output(torch.as_tensor(logits))
        probs = self._probabilities_from_logits(logits)[0]
        k = min(topk, probs.numel())
        top_probs, top_indices = probs.topk(k)

        display_image = self._prepare_image_for_display(image)
        draw = ImageDraw.Draw(display_image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        lines = [f"{label_names.get(int(idx), str(int(idx)))}: {float(prob):.3f}" for idx, prob in zip(top_indices, top_probs)]
        text = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        pad = 6
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(0, 0, 0),
        )
        draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font, spacing=4)
        display_image.save(save_path)

    def test_model(
        self,
        test_loader,
        device: str | torch.device | None = None,
        output_dir: str | Path = "predictions_classification",
        label_names: dict[int, str] | None = None,
        max_images: int = 12,
    ):
        model = self.ensure_model()
        device = torch.device(device or self.device)
        model.eval()
        output_dir = self.ensure_dir(output_dir)
        label_names = label_names or self.label_names or DEFAULT_LABEL_NAMES

        saved_images = 0
        with torch.inference_mode():
            for step, (images, targets) in enumerate(test_loader, start=1):
                images = images.to(device)
                logits = model(images)
                predictions = self._predict_from_logits(logits.detach())

                for image_idx in range(images.shape[0]):
                    image = images[image_idx].detach().cpu()
                    pred = int(predictions[image_idx].item())
                    gt = int(targets[image_idx].item()) if torch.is_tensor(targets) else int(targets[image_idx])
                    save_path = output_dir / f"pred_{step:04d}_{image_idx:02d}.png"
                    self.visualize_prediction(
                        image=image,
                        output=logits[image_idx].detach().cpu(),
                        save_path=save_path,
                        label_names=label_names,
                    )
                    print(
                        f"saved {save_path} | pred={label_names.get(pred, str(pred))} | "
                        f"gt={label_names.get(gt, str(gt))}"
                    )
                    saved_images += 1
                    if saved_images >= max_images:
                        return

    def export_onnx(
        self,
        onnx_path: str | Path = "classification_pt.onnx",
        input_size: tuple[int, int] = (224, 224),
        state_dict_path: str | Path | None = None,
        num_classes: int = 2,
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
        dummy_input = torch.rand((1, 3, input_size[0], input_size[1]), device=self.device)
        onnx_path = Path(onnx_path)

        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch"},
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
        criterion: torch.nn.Module | str | None = "cross_entropy",
        output_dim: int | None = None,
        pretrained_backbone: bool = False,
        initial_weights_path: str | Path | None = None,
        save_checkpoints: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)

        resolved_output_dim = output_dim
        if resolved_output_dim is None:
            if isinstance(criterion, str):
                is_binary_criterion = criterion.lower().strip() in {"bce", "bce_logits"}
            else:
                is_binary_criterion = False
            resolved_output_dim = 1 if is_binary_criterion else num_classes

        self.criterion = self._resolve_criterion(criterion)

        if self.model is None or self.num_classes != num_classes or self.output_dim != resolved_output_dim:
            self.build_model(
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
                output_dim=resolved_output_dim,
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

        if tune.get_checkpoint():
            loaded_checkpoint = tune.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                )
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=2)

        for epoch in range(1, epochs + 1):
            avg_loss, train_metrics = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                criterion=self.criterion,
            )
            self.history.append(avg_loss)

            # train_log = self.format_loss_components({"loss": avg_loss, **train_metrics})
            # print(f"Epoch {epoch} finished | {train_log}")

            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate_model(eval_loader=eval_loader, criterion=criterion)
                if save_checkpoints:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                        torch.save(
                            (self.model.state_dict(), self.optimizer.state_dict()), path
                        )
                        checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
                        tune.report(eval_metrics, checkpoint=checkpoint)
                else:
                    tune.report(eval_metrics)
                
            # print("=================")

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
        criterion: torch.nn.Module | str | None = "cross_entropy",
        output_dim: int | None = None,
        pretrained_backbone: bool = False,
        initial_weights_path: str | Path | None = None,
        save_checkpoints: bool = False,
        resume_path: str | Path | None = None,
        cpus_per_trial: int = 2,
        gpus_per_trial: int = 1,
        max_num_epochs: int | None = None,
        grace_period: int = 1,
        num_trials: int = 10
    ):
        if resume_path is not None and not save_checkpoints:
            raise ValueError("resume_path requires save_checkpoints=True so resumed trials can continue writing checkpoints.")
        
        if max_num_epochs is None:
            max_num_epochs = epochs

        tune_scheduler = schedulers.ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=2
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
                    criterion=criterion,
                    output_dim=output_dim,
                    pretrained_backbone=pretrained_backbone,
                    initial_weights_path=initial_weights_path,
                    save_checkpoints=save_checkpoints),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
            )
        tune_config = tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=tune_scheduler,
                num_samples=num_trials
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

        best_result = results.get_best_result("loss", "min")

        print(f"Best trial config: {best_result.config}")
        print(f"Best trial validation loss: {best_result.metrics['loss']}")
        print(f"Best trial validation accuracy: {best_result.metrics['accuracy']}")
        print(f"Best trial validation class accuracy: {best_result.metrics['class_accuracy']}")
        print(f"Best trial validation num_samples: {best_result.metrics['num_samples']}")


def _task_from_model(model, device=None, label_names=None):
    return Classification(device=device, model=model, label_names=label_names)


def build_model(num_classes: int = 2, pretrained_backbone: bool = False, output_dim: int | None = None):
    return Classification().build_model(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        output_dim=output_dim,
    )


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
    num_classes=2,
    criterion="cross_entropy",
    output_dim=None,
    pretrained_backbone=False,
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
        num_classes=num_classes,
        criterion=criterion,
        output_dim=output_dim,
        pretrained_backbone=pretrained_backbone,
    )


def evaluate_model(model, test_loader, device, label_names=None, criterion="cross_entropy"):
    return _task_from_model(model, device=device, label_names=label_names).evaluate_model(
        eval_loader=test_loader,
        device=device,
        label_names=label_names,
        criterion=criterion,
    )


def visualize_prediction(image, output, save_path, label_names=None, topk=3):
    return _task_from_model(None, label_names=label_names).visualize_prediction(
        image=image,
        output=output,
        save_path=save_path,
        label_names=label_names,
        topk=topk,
    )


def test_model(
    model,
    test_loader,
    device,
    output_dir="predictions_classification",
    label_names=None,
    max_images=12,
):
    return _task_from_model(model, device=device, label_names=label_names).test_model(
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        label_names=label_names,
        max_images=max_images,
    )


def export_onnx(
    model=None,
    onnx_path="classification_pt.onnx",
    input_size=(224, 224),
    state_dict_path=None,
    num_classes=2,
    pretrained_backbone=False,
):
    return _task_from_model(model).export_onnx(
        onnx_path=onnx_path,
        input_size=input_size,
        state_dict_path=state_dict_path,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )

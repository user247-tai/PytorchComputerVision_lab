from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.functional import to_pil_image


class PytorchVisionLab(ABC):
    """Shared lifecycle for PyTorch vision tasks."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
        label_names: dict[int, str] | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = model
        self.label_names = label_names or {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
        self.criterion: torch.nn.Module | None = None
        self.history: list[float] = []
        self.eval_history: list[dict[str, Any]] = []
        self.trained_model_path: Path | None = None

    @staticmethod
    def move_to_device(value: Any, device: str | torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: PytorchVisionLab.move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(PytorchVisionLab.move_to_device(item, device) for item in value)
        return value

    @staticmethod
    def freeze_batchnorm_layers(model: torch.nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
        image = image.detach().cpu()
        if image.dtype != torch.uint8:
            image = (image.clamp(0, 1) * 255).to(torch.uint8)
        return image

    @staticmethod
    def save_tensor_image(image: torch.Tensor, save_path: str | Path) -> None:
        to_pil_image(PytorchVisionLab.image_to_uint8(image)).save(save_path)

    @staticmethod
    def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def save_model_state_dict(path: str | Path, model: torch.nn.Module) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        return path

    @staticmethod
    def load_model_state_dict(
        model: torch.nn.Module,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        state_dict = torch.load(path, map_location=map_location)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def format_loss_components(loss_components: dict[str, float]) -> str:
        if not loss_components:
            return ""
        return " | ".join(f"{name}={value:.4f}" for name, value in sorted(loss_components.items()))

    @staticmethod
    def flatten_numeric_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
        flattened: dict[str, float] = {}
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}"
            if isinstance(value, dict):
                flattened.update(PytorchVisionLab.flatten_numeric_metrics(value, prefix=f"{metric_name}_"))
            elif torch.is_tensor(value) and value.numel() == 1:
                flattened[metric_name] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float)):
                flattened[metric_name] = float(value)
        return flattened

    @staticmethod
    def compute_average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
        if recall.numel() == 0:
            return 0.0

        mrec = torch.cat(
            [torch.tensor([0.0], dtype=recall.dtype), recall, torch.tensor([1.0], dtype=recall.dtype)]
        )
        mpre = torch.cat(
            [torch.tensor([0.0], dtype=precision.dtype), precision, torch.tensor([0.0], dtype=precision.dtype)]
        )

        for idx in range(mpre.numel() - 2, -1, -1):
            mpre[idx] = torch.maximum(mpre[idx], mpre[idx + 1])

        changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
        return float(ap)

    def ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            self.model = self.build_model()
        self.model = self.model.to(self.device)
        return self.model

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("Model has not been built yet.")

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_name = optimizer_name.lower().strip()
        optimizer_kwargs = dict(optimizer_kwargs or {})

        optimizer_factories = {
            "sgd": (torch.optim.SGD, {"momentum": 0.9}),
            "adam": (torch.optim.Adam, {"betas": (0.9, 0.999), "eps": 1e-8}),
            "adamw": (torch.optim.AdamW, {"betas": (0.9, 0.999), "eps": 1e-8}),
            "rmsprop": (torch.optim.RMSprop, {"momentum": 0.0, "alpha": 0.99, "eps": 1e-8}),
        }

        if optimizer_name not in optimizer_factories:
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. Supported optimizers: {', '.join(sorted(optimizer_factories))}"
            )

        optimizer_cls, defaults = optimizer_factories[optimizer_name]
        kwargs = {**defaults, **optimizer_kwargs}
        kwargs.update({"lr": lr, "weight_decay": weight_decay})
        return optimizer_cls(params, **kwargs)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        # Default: no scheduler. Task classes can override this when needed.
        return None

    def train_model(
        self,
        train_loader,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        checkpoint_path: str | Path | None = None,
        trained_model_path: str | Path | None = None,
        eval_loader=None,
        optimizer_name: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        **train_kwargs,
    ):
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

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_one_epoch(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                **train_kwargs,
            )
            self.history.append(avg_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate_model(eval_loader=eval_loader, **train_kwargs)
                self.eval_history.append(eval_metrics)

            if checkpoint_path is not None:
                payload = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "avg_loss": avg_loss,
                    "history": list(self.history),
                }
                if eval_metrics is not None:
                    payload["eval_metrics"] = eval_metrics
                    payload["eval_history"] = list(self.eval_history)
                self.save_checkpoint(checkpoint_path, payload)

        if trained_model_path is not None:
            self.trained_model_path = self.save_model_state_dict(trained_model_path, self.ensure_model())

        return self.model, self.history, self.eval_history

    @abstractmethod
    def build_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def visualize_prediction(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def test_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def optimize_parameters(self, *args, **kwargs):
        raise NotADirectoryError

    @abstractmethod
    def trial(self, *args, **kwargs):
        raise NotADirectoryError

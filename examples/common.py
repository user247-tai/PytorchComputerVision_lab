from __future__ import annotations

from pathlib import Path

import torch


def resolve_weights(weights_path: str | Path | None):
    if weights_path is None:
        return None
    weights = torch.load(weights_path, map_location="cpu")
    if isinstance(weights, dict) and "model_state_dict" in weights:
        return weights["model_state_dict"]
    return weights


def load_weights_into_model(model: torch.nn.Module, weights_path: str | Path | None) -> torch.nn.Module:
    state_dict = resolve_weights(weights_path)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def device_from_arg(device: str | None) -> str:
    return device or ("cuda:0" if torch.cuda.is_available() else "cpu")

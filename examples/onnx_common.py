from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image


def create_session(model_path: str | Path, providers: Sequence[str] | None = None) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    if providers is None:
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in preferred_providers if provider in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path_or_bytes=str(model_path), providers=list(providers))


def load_image_tensor(image_path: str | Path, size: int | tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    if isinstance(size, int):
        size = (size, size)

    resized = image.resize(size, Image.Resampling.BILINEAR)
    image_tensor = torch.from_numpy(np.array(resized, dtype=np.uint8)).permute(2, 0, 1).contiguous()
    input_tensor = image_tensor.float().div(255.0).unsqueeze(0)
    return image_tensor, input_tensor


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)
    return image

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from examples.onnx_common import create_session, load_image_tensor


DEFAULT_MODEL = Path("classification_pt.onnx")
DEFAULT_IMAGE = Path("data/classification/dataset/test/class1/example.png")
DEFAULT_SAVE = Path("classification_onnx_pred.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime on the exported classification model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--save", default=str(DEFAULT_SAVE))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def visualize_topk(image: torch.Tensor, probs: torch.Tensor, label_names: dict[int, str], save_path: str, topk: int) -> None:
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)

    pil_image = Image.fromarray(image.permute(1, 2, 0).contiguous().numpy())
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    topk = min(topk, probs.numel())
    values, indices = probs.topk(topk)
    lines = [f"{label_names.get(int(idx), str(int(idx)))}: {float(val):.3f}" for val, idx in zip(values, indices)]
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    pad = 6
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
    draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font, spacing=4)
    pil_image.save(save_path)


def main() -> None:
    args = parse_args()
    session = create_session(args.model)
    print("Providers:", session.get_providers())

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    print("Input:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
    for output in session.get_outputs():
        print("Output:", output.name, output.shape)

    image_tensor, input_tensor = load_image_tensor(args.image, args.size)
    outputs = session.run(output_names=output_names, input_feed={input_name: input_tensor.numpy()})
    if not outputs:
        raise RuntimeError("The ONNX model did not return any outputs.")

    logits = torch.from_numpy(np.asarray(outputs[0]))
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits[0], dim=0)
    label_names = {idx: f"class_{idx}" for idx in range(probs.numel())}
    pred_idx = int(probs.argmax().item())

    visualize_topk(image_tensor, probs, label_names, args.save, args.topk)
    print(
        f"Classification prediction saved to {args.save} | "
        f"pred={label_names.get(pred_idx, str(pred_idx))} | probs_shape={tuple(probs.shape)}"
    )


if __name__ == "__main__":
    main()

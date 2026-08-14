from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from examples.onnx_common import create_session, image_to_uint8, load_image_tensor


DEFAULT_MODEL = Path("detection_pt.onnx")
DEFAULT_IMAGE = Path("fire_smoke.jpg")
DEFAULT_SAVE = Path("detection_onnx_pred.png")
DEFAULT_LABELS = {1: "fire", 2: "smoke"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime on the exported detection model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--save", default=str(DEFAULT_SAVE))
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--size", type=int, default=480)
    return parser.parse_args()


def decode_outputs(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, conf: float):
    boxes = torch.from_numpy(np.asarray(boxes)).float()
    labels = torch.from_numpy(np.asarray(labels)).long()
    scores = torch.from_numpy(np.asarray(scores)).float()

    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)

    keep = scores >= conf
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
    }


def visualize_predictions(image: torch.Tensor, predictions: dict[str, torch.Tensor], save_path: str) -> None:
    image = image_to_uint8(image)
    boxes = predictions["boxes"].detach().cpu()
    labels = predictions["labels"].detach().cpu()
    scores = predictions["scores"].detach().cpu()

    if boxes.numel() == 0:
        to_pil_image(image).save(save_path)
        print(f"No detections found. Saved input image to {save_path}")
        return

    label_texts = [
        f"{DEFAULT_LABELS.get(int(label_id), str(int(label_id)))}: {float(score):.3f}"
        for label_id, score in zip(labels, scores)
    ]
    palette = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    box_colors = [palette[i % len(palette)] for i in range(boxes.shape[0])]
    drawn = draw_bounding_boxes(image, boxes.round().to(torch.int64), labels=label_texts, colors=box_colors, width=3)
    to_pil_image(drawn).save(save_path)
    print(f"Saved visualization to {save_path}")


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

    if len(outputs) != 3:
        raise RuntimeError(f"Expected 3 outputs from the detection model, got {len(outputs)}")

    predictions = decode_outputs(outputs[0], outputs[1], outputs[2], args.conf)
    print(
        f"Detections kept: {predictions['boxes'].shape[0]} | "
        f"boxes={tuple(predictions['boxes'].shape)} | "
        f"labels={tuple(predictions['labels'].shape)} | "
        f"scores={tuple(predictions['scores'].shape)}"
    )
    visualize_predictions(image_tensor, predictions, args.save)


if __name__ == "__main__":
    main()

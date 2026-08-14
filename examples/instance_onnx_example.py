from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from examples.onnx_common import create_session, image_to_uint8, load_image_tensor


DEFAULT_MODEL = Path("instance_pt.onnx")
DEFAULT_IMAGE = Path("data/segmentation/instance/pennfudan_coco/test/PennPed00066.png")
DEFAULT_SAVE = Path("instance_onnx_pred.png")
DEFAULT_LABELS = {1: "person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime on the exported instance segmentation model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--save", default=str(DEFAULT_SAVE))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mask-thres", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=480)
    return parser.parse_args()


def _decode_maskrcnn_outputs(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    conf_threshold: float,
    mask_threshold: float,
):
    boxes_t = torch.from_numpy(np.asarray(boxes)).float()
    labels_t = torch.from_numpy(np.asarray(labels)).long()
    scores_t = torch.from_numpy(np.asarray(scores)).float()
    masks_t = torch.from_numpy(np.asarray(masks))

    if boxes_t.ndim == 1:
        boxes_t = boxes_t.reshape(-1, 4)
    if masks_t.ndim == 2:
        masks_t = masks_t.unsqueeze(0)
    if masks_t.ndim == 4:
        masks_t = masks_t[:, 0]

    keep = scores_t > conf_threshold
    boxes_t = boxes_t[keep]
    labels_t = labels_t[keep]
    scores_t = scores_t[keep]
    masks_t = masks_t[keep]

    if masks_t.dtype != torch.bool:
        masks_t = masks_t > mask_threshold

    return {
        "boxes": boxes_t,
        "labels": labels_t,
        "scores": scores_t,
        "masks": masks_t,
    }


def _decode_yolo_style_outputs(
    output0: np.ndarray,
    output1: np.ndarray,
    input_size: int,
    conf_threshold: float,
    mask_threshold: float,
):
    pred = output0[0]
    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_ids = pred[:, 5].astype(np.int64)
    mask_coeffs = pred[:, 6:]

    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if boxes.shape[0] == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "masks": torch.empty((0, input_size, input_size), dtype=torch.bool),
        }

    protos = output1[0]
    proto_h, proto_w = protos.shape[1], protos.shape[2]
    protos_flat = protos.reshape(protos.shape[0], -1)

    masks = mask_coeffs @ protos_flat
    masks = 1.0 / (1.0 + np.exp(-masks))
    masks = masks.reshape(-1, proto_h, proto_w)

    masks_ts = torch.from_numpy(masks).unsqueeze(1).float()
    masks_ts = F.interpolate(masks_ts, size=(input_size, input_size), mode="bilinear", align_corners=False)
    masks_ts = masks_ts.squeeze(1) > mask_threshold

    return {
        "boxes": torch.from_numpy(boxes).float(),
        "scores": torch.from_numpy(scores).float(),
        "labels": torch.from_numpy(class_ids).long(),
        "masks": masks_ts,
    }


def decode_outputs(outputs, input_size: int, conf_threshold: float, mask_threshold: float):
    if len(outputs) >= 4 and np.asarray(outputs[0]).ndim <= 2:
        return _decode_maskrcnn_outputs(
            boxes=outputs[0],
            labels=outputs[1],
            scores=outputs[2],
            masks=outputs[3],
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
        )

    if len(outputs) >= 2:
        return _decode_yolo_style_outputs(
            output0=outputs[0],
            output1=outputs[1],
            input_size=input_size,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
        )

    raise RuntimeError(f"Unsupported instance segmentation ONNX outputs: {len(outputs)} tensors")


def visualize_predictions(image: torch.Tensor, predictions: dict[str, torch.Tensor], save_path: str) -> None:
    image = image_to_uint8(image)
    boxes = predictions["boxes"].detach().cpu()
    scores = predictions["scores"].detach().cpu()
    labels = predictions["labels"].detach().cpu()
    masks = predictions["masks"].detach().cpu()

    if boxes.numel() == 0:
        to_pil_image(image).save(save_path)
        print(f"No detections found. Saved input image to {save_path}")
        return

    label_texts = [
        f"{DEFAULT_LABELS.get(int(label_id), str(int(label_id)))}: {float(score):.3f}"
        for label_id, score in zip(labels, scores)
    ]
    palette = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    mask_colors = [palette[i % len(palette)] for i in range(masks.shape[0])]
    box_colors = [palette[i % len(palette)] for i in range(boxes.shape[0])]

    drawn = draw_segmentation_masks(image, masks, alpha=0.5, colors=mask_colors)
    drawn = draw_bounding_boxes(drawn, boxes.round().to(torch.int64), labels=label_texts, colors=box_colors, width=3)
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
    predictions = decode_outputs(outputs, input_size=args.size, conf_threshold=args.conf, mask_threshold=args.mask_thres)

    print(
        f"Detections kept: {predictions['boxes'].shape[0]} | "
        f"boxes={tuple(predictions['boxes'].shape)} | "
        f"masks={tuple(predictions['masks'].shape)}"
    )
    visualize_predictions(image_tensor, predictions, args.save)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from examples.onnx_common import create_session, load_image_tensor
from vision_tasks.semantic_segmentation import (
    BACKGROUND_INDEX,
    model_output_to_bool_masks,
    save_colorized_label_map,
    visualize_mask_overlay,
)


DEFAULT_MODEL = Path("semantic_segmentation_pt.onnx")
DEFAULT_IMAGE = Path("data/segmentation/semantic/dataset1/test/images/0016E5_08113.png")
DEFAULT_SAVE = Path("semantic_onnx_pred.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime on the exported semantic segmentation model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--save", default=str(DEFAULT_SAVE))
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--background-index", type=int, default=BACKGROUND_INDEX)
    return parser.parse_args()


def select_main_logits(outputs: list[np.ndarray], output_names: list[str]) -> np.ndarray:
    if not outputs:
        raise RuntimeError("The ONNX model did not return any outputs.")

    output_map = {name: value for name, value in zip(output_names, outputs)}
    if "out" in output_map:
        return output_map["out"]
    if "output" in output_map:
        return output_map["output"]

    non_aux_names = [name for name in output_names if "aux" not in name.lower()]
    if non_aux_names:
        return output_map[non_aux_names[0]]

    return outputs[0]


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

    logits = torch.from_numpy(np.asarray(select_main_logits(outputs, output_names)))
    pred_mask, class_ids = model_output_to_bool_masks(logits, background_index=args.background_index)
    pred_label_map = logits.argmax(dim=1)[0] if logits.ndim == 4 else logits.argmax(dim=0)

    visualize_mask_overlay(
        image=image_tensor,
        masks=pred_mask,
        class_ids=class_ids,
        save_path=args.save,
    )
    save_colorized_label_map(pred_label_map, Path(args.save).with_name(Path(args.save).stem + "_labels.png"))

    print(
        f"Semantic prediction saved to {args.save} | "
        f"class_ids={class_ids} | logits_shape={tuple(logits.shape)}"
    )


if __name__ == "__main__":
    main()

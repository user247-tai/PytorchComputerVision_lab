# Computer Vision Lab

A PyTorch computer vision project with reusable task classes, dataset loaders, training and inference examples, ONNX export, and dataset conversion utilities. It currently supports object detection, instance segmentation, semantic segmentation, and multi-class or binary image classification.

## Project structure

```text
datasets/       Dataset implementations and batching helpers
examples/       Train, validate, test, tune, export, and ONNX inference scripts
helper/         Transforms, COCO evaluation, and shared utilities
tools/          Dataset conversion scripts
vision_tasks/   Task implementations and shared PyTorch lifecycle
requirements.txt
```

## Install

Use Python with compatible PyTorch and Torchvision builds for your platform, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include PyTorch, Torchvision, ONNX Runtime, Ray Tune, NumPy, Pillow, OpenCV, and pycocotools.

## Task API

`vision_tasks.base.PytorchVisionLab` provides shared model training, evaluation, testing, prediction visualization, and ONNX export methods. The task classes are:

```python
from vision_tasks import Classification, Detection, InstanceSegmentation, SemanticSegmentation
```

Dataset classes and collate functions are exported from `datasets`:

```python
from datasets import (
    CocoDetectionDataset,
    COCOInstanceDataset,
    ImageBinaryClassificationDataset,
    ImageClassificationDataset,
    SemanticSegmentationDataset,
    detection_collate_fn,
    instance_collate_fn,
)
```

## Dataset formats

Pass the dataset root with `--root`. Example scripts default to the split names `train`, `valid`, and `test`; use `--train-split`, `--eval-split`, and `--test-split` to change them.

### Detection and instance segmentation

Both use COCO annotation JSON with this layout:

```text
dataset/
  train/                 # Images referenced by the annotation JSON
  valid/
  test/
  annotations/
    instances_train.json
    instances_valid.json
    instances_test.json
```

Detection loads boxes and labels; instance segmentation also loads masks. Category IDs are mapped to consecutive labels starting at 1 by default. Torchvision models reserve label 0 for background, so set `num_classes` to the foreground class count plus one when building a model. Detection and instance datasets use `detection_collate_fn` and `instance_collate_fn` respectively, since each image may have a different number of objects.

### Semantic segmentation

Images and indexed mask images are paired by filename stem:

```text
dataset/
  train/images/   train/masks/
  valid/images/   valid/masks/
  test/images/    test/masks/
```

Masks should store class IDs as pixel values and use the `.png` suffix by default. `--num-classes 0` (the default) infers the class count from the maximum mask label across the configured splits. For example, labels from 0 through 11 require 12 classes.

### Classification

Each split contains one subfolder per class:

```text
dataset/
  train/class_a/  train/class_b/
  valid/class_a/  valid/class_b/
  test/class_a/   test/class_b/
```

Class indices are derived from the training folder names and reused for evaluation and test. `--num-classes 0` infers the count. Select `--dataset-type binary` for the two-class dataset implementation; `--criterion` accepts `cross_entropy`, `bce`, or `bce_logits`. BCE criteria use a one-logit model output.

## Training and evaluation examples

Each task example accepts `--mode train|valid|test|export|tune|all`. `all` runs training, validation, testing, and export in sequence. The `tune` mode runs Ray Tune hyperparameter search. Common options include `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--optimizer` (`sgd`, `adam`, `adamw`, or `rmsprop`), `--weights`, `--trained-model-dir`, and `--trained-model-name`. Use `--help` on a script for all task-specific options. Model checkpoints, trained weights, ONNX files, and prediction outputs are written to the configured paths.

Train and validate detection, then run the complete workflow:

```bash
python examples/detection_example.py --mode train --root data/detection/my_coco
python examples/detection_example.py --mode all --root data/detection/my_coco --epochs 10
```

Instance segmentation:

```bash
python examples/instance_segmentation_example.py --mode all --root data/segmentation/instance/my_coco --epochs 20
```

Semantic segmentation:

```bash
python examples/semantic_segmentation_example.py --mode all --root data/segmentation/semantic/my_dataset --num-classes 12
```

Classification:

```bash
python examples/classification_example.py --mode all --root data/classification/my_dataset --epochs 20
python examples/classification_example.py --mode train --dataset-type binary --criterion bce_logits --root data/classification/binary_dataset
```

For a tuning run, use `--mode tune`; Ray settings include `--num-samples`, `--gpus-per-trial`, `--grace-period`, `--resume-path`, and `--save-checkpoints`.

## ONNX Runtime inference

The task scripts export ONNX models with `--mode export` or as the final step of `--mode all`. The matching inference examples accept `--model` and `--image` (and an optional `--save` output path):

```bash
python examples/detection_onnx_example.py --model detection_pt.onnx --image path/to/image.jpg
python examples/instance_onnx_example.py --model instance_segmentation_pt.onnx --image path/to/image.jpg
python examples/semantic_onnx_example.py --model semantic_segmentation_pt.onnx --image path/to/image.png
python examples/classification_onnx_example.py --model classification_pt.onnx --image path/to/image.jpg
```

Detection and instance scripts expose confidence thresholds; classification exposes `--topk`, and the segmentation scripts accept input-size options. The semantic inference script uses the main model output when an auxiliary output is also present. Instance inference supports the output forms handled by its decoder, including Mask R-CNN and YOLO-style outputs.

## Dataset conversion tools

The `tools/` directory contains:

- `pennfudan_to_coco.py` — convert PennFudan pedestrian data to COCO instance annotations; supports split ratios, a random seed, and `--no-copy-images`.
- `coco2yolo.py` and `yolo2coco.py` — convert bounding-box datasets between COCO and YOLO formats. Edit the dataset paths and class definitions in the scripts before running; these two scripts currently use in-file configuration.
- `semantic_png_to_yolo.py` — prepare YOLO semantic segmentation data from indexed PNG masks.
- `semantic_mask_to_yolo_seg.py` — convert indexed masks into YOLO segmentation labels, treating connected components as instances.

The semantic conversion tools accept `--src-root`, `--output-root`, `--val-ratio`, and `--seed`. Run a tool with `--help` to see its options.

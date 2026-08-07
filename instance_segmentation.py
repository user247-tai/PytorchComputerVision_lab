import os
from pathlib import Path

import torch
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.mobilenet import MobileNet_V2_Weights, MobileNet_V3_Large_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, Resize, ToDtype, ToImage, ToPureTensor
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from datasets import COCOInstanceDataset, instance_collate_fn


INPUT_SIZE = 512


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def freeze_model_backbone(model) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()


def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    # num_classes includes background, so PennFudanPed uses 2: background + person.
    trainable_layers = 0 if freeze_backbone else 2
    backbone = mobilenet_backbone(
        backbone_name="mobilenet_v3_large",
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        fpn=True,
        trainable_layers=trainable_layers,
    )

    anchor_generator = AnchorGenerator(
        sizes=(
            (16, 32),     
            (64, 128),   
            (256, 512)  
        ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,  # 3 tỷ lệ khung hình (1:2, 1:1, 2:1) cho cả 3 tầng
    )
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "pool"], output_size=7, sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "pool"], output_size=14, sampling_ratio=2)
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=300,
        max_size=500,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_nms_thresh=0.3
    )
    if freeze_backbone:
        freeze_model_backbone(model)
    return model


def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(model, loader, optimizer, device, epoch, warmup_scheduler=None):
    model.train()
    running_loss = 0.0
    running_loss_components = {}

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [move_to_device(target, device) for target in targets]

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

    num_batches = max(len(loader), 1)
    avg_loss = running_loss / num_batches
    avg_loss_components = {
        name: value / num_batches for name, value in sorted(running_loss_components.items())
    }
    return avg_loss, avg_loss_components


def train_model(
    model,
    train_loader,
    device,
    epochs=10,
    lr=0.005,
    weight_decay=0.0005,
    checkpoint_path=None,
    eval_loader=None,
):
    model.to(device)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    history = []
    eval_history = []
    final_loss_components = {}
    for epoch in range(1, epochs + 1):
        warmup_scheduler = None
        if epoch == 1:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(train_loader) - 1)
            if warmup_iters > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=warmup_factor,
                    total_iters=warmup_iters,
                )

        avg_loss, avg_loss_components = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            warmup_scheduler=warmup_scheduler,
        )
        scheduler.step()
        history.append(avg_loss)
        final_loss_components = avg_loss_components
        component_log = " | ".join(
            f"{name}={value:.4f}" for name, value in avg_loss_components.items()
        )
        print(f"Epoch {epoch} finished | {component_log} | loss={avg_loss:.4f}")

        eval_metrics = None
        if eval_loader is not None:
            eval_metrics = evaluate_model(
                model=model,
                test_loader=eval_loader,
                device=device,
                score_threshold=0.5,
            )
            eval_history.append(eval_metrics)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_loss": avg_loss,
                    "history": history,
                    "eval_metrics": eval_metrics,
                    "eval_history": eval_history,
                },
                checkpoint_path,
            )

        print("====================")

    return model, history, eval_history, final_loss_components


def compute_average_precision(recall, precision):
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


def mask_iou(pred_mask: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    pred_mask = pred_mask.to(torch.bool).reshape(-1)
    if gt_masks.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    gt_masks = gt_masks.to(torch.bool).reshape(gt_masks.shape[0], -1)
    intersection = torch.logical_and(gt_masks, pred_mask.unsqueeze(0)).sum(dim=1).to(torch.float32)
    union = torch.logical_or(gt_masks, pred_mask.unsqueeze(0)).sum(dim=1).to(torch.float32)
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def evaluate_class_at_iou(predictions, ground_truths, class_id, iou_threshold, score_threshold=None):
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

        ious = mask_iou(pred_mask, gt_masks)
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
        ap = compute_average_precision(recall, precision)
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


def evaluate_model(
    model,
    test_loader,
    device,
    score_threshold=0.5,
    label_names=None,
):
    del score_threshold, label_names

    model.to(device)
    was_training = model.training
    model.eval()

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    coco = get_coco_api_from_dataset(test_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])

    total_loss = 0.0
    total_images = 0
    running_loss_components = {}

    for images, targets in test_loader:
        images_device = [image.to(device) for image in images]
        targets_device = [move_to_device(target, device) for target in targets]

        model.train()
        freeze_batchnorm_layers(model)
        loss_dict = model(images_device, targets_device)
        batch_loss = sum(loss for loss in loss_dict.values())

        total_loss += float(batch_loss.detach().cpu()) * len(images_device)
        total_images += len(images_device)
        for name, value in loss_dict.items():
            running_loss_components[name] = running_loss_components.get(name, 0.0) + float(value.detach().cpu()) * len(images_device)

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
    avg_loss_components = {
        name: value / max(total_images, 1) for name, value in sorted(running_loss_components.items())
    }
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

    component_log = " | ".join(f"{name}={value:.4f}" for name, value in avg_loss_components.items())
    print(f"Test loss={metrics['loss']:.4f} | {component_log}")
    return metrics

def visualize_prediction(
    image: torch.Tensor,
    output: dict,
    score_threshold: float,
    save_path: str,
    label_names=None,
):
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)

    if label_names is None:
        label_names = {1: "person"}

    keep = output["scores"].detach().cpu() > score_threshold
    boxes = output["boxes"].detach().cpu()[keep]
    labels = output["labels"].detach().cpu()[keep]
    scores = output["scores"].detach().cpu()[keep]
    masks = output["masks"].detach().cpu()[keep, 0] > 0.5

    if boxes.numel() > 0:
        label_texts = [
            f"{label_names.get(int(label_id), str(int(label_id)))}: {float(score):.3f}"
            for label_id, score in zip(labels, scores)
        ]
        drawn = draw_segmentation_masks(image, masks, alpha=0.5, colors="red")
        drawn = draw_bounding_boxes(drawn, boxes.long(), labels=label_texts, colors="green", width=3)
    else:
        drawn = image

    to_pil_image(drawn).save(save_path)


def test_model(
    model,
    test_loader,
    device,
    output_dir="predictions_instance",
    score_threshold=0.5,
    label_names=None,
    max_images=12,
):
    model.to(device)
    model.eval()

    if label_names is None:
        label_names = {1: "person"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(test_loader, start=1):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for image_idx, (image, output) in enumerate(zip(images, outputs)):
                out_path = output_dir / f"pred_{step:04d}_{image_idx:02d}.png"
                visualize_prediction(
                    image=image,
                    output=output,
                    score_threshold=score_threshold,
                    save_path=str(out_path),
                    label_names=label_names,
                )
                print(f"saved {out_path}")
                saved_images += 1

                if saved_images >= max_images:
                    return


def main() -> None:
    transforms = Compose(
        [
            RandomHorizontalFlip(0.5),
            ToDtype(torch.float, scale=True),
            ToPureTensor(),
        ]
    )

    valid_transforms = Compose(
        [
            ToDtype(torch.float, scale=True),
            ToPureTensor(),
        ]
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root = "/home/server/computer_vision_lab/data/segmentation/instance/pennfudan_coco"

    train_dataset = COCOInstanceDataset(
        images_root=os.path.join(root, "train"),
        annotation_path=os.path.join(root, "annotations", "instances_train.json"),
        transforms=transforms,
    )

    test_dataset = COCOInstanceDataset(
        images_root=os.path.join(root, "test"),
        annotation_path=os.path.join(root, "annotations", "instances_test.json"),
        transforms=valid_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=instance_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=instance_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=2)

    # model_state_dict = torch.load("/home/server/computer_vision_lab/maskrcnn_pennfudan_cc.pt")
    # model.load_state_dict(model_state_dict)

    model, history, eval_history, final_loss_components = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=20,
        checkpoint_path="checkpoints/maskrcnn_pennfudan.pt",
        eval_loader=test_loader,
    )

    final_component_log = " | ".join(
        f"{name}={value:.4f}" for name, value in final_loss_components.items()
    )
    print(f"Training complete. {final_component_log} | Final loss: {history[-1]:.4f}")
    torch.save(model.state_dict(), "maskrcnn_pennfudan_cc.pt")

    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir="predictions_instance",
        score_threshold=0.5,
    )


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage, RandomHorizontalFlip
from torchvision.utils import draw_bounding_boxes
from torchinfo import summary

from datasets import CocoDetectionDataset, detection_collate_fn


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(losses.detach().cpu())
        running_loss += loss_value
        print(
            f"Epoch {epoch} | Step {step}/{len(loader)} | "
            f"loss={loss_value:.4f} | total_loss={running_loss / step:.4f}"
        )

    return running_loss / max(len(loader), 1)


def train_model(
    model,
    train_loader,
    device,
    epochs=10,
    lr=0.001,
    weight_decay=0.0005,
    checkpoint_path=None,
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
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        history.append(avg_loss)
        print(f"Epoch {epoch} finished | avg_loss={avg_loss:.4f}")

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_loss": avg_loss,
                    "history": history,
                },
                checkpoint_path,
            )

    return model, history


def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


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


def evaluate_class_at_iou(predictions, ground_truths, class_id, iou_threshold, score_threshold=None):
    gt_boxes_by_image = []
    matched_flags_by_image = []
    total_gt = 0

    for gt in ground_truths:
        gt_mask = gt["labels"] == class_id
        gt_boxes = gt["boxes"][gt_mask].float()
        gt_boxes_by_image.append(gt_boxes)
        matched_flags_by_image.append(torch.zeros(len(gt_boxes), dtype=torch.bool))
        total_gt += len(gt_boxes)

    pred_records = []
    for image_idx, pred in enumerate(predictions):
        pred_mask = pred["labels"] == class_id
        pred_boxes = pred["boxes"][pred_mask].float()
        pred_scores = pred["scores"][pred_mask].float()

        if score_threshold is not None:
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

        for box, score in zip(pred_boxes, pred_scores):
            pred_records.append((image_idx, float(score), box))

    pred_records.sort(key=lambda item: item[1], reverse=True)

    tp = []
    fp = []
    matched_ious = []

    for image_idx, _, pred_box in pred_records:
        gt_boxes = gt_boxes_by_image[image_idx]
        if gt_boxes.numel() == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
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
    model.to(device)
    was_training = model.training
    model.eval()

    if label_names is None:
        label_names = {1: "fire", 2: "smoke"}

    predictions = []
    ground_truths = []
    total_loss = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, targets in test_loader:
            images_device = [image.to(device) for image in images]
            targets_device = [move_to_device(target, device) for target in targets]

            model.train()
            freeze_batchnorm_layers(model)
            loss_dict = model(images_device, targets_device)
            batch_loss = sum(loss for loss in loss_dict.values())
            total_loss += float(batch_loss.detach().cpu()) * len(images_device)
            model.eval()

            outputs = model(images_device)

            for output, target in zip(outputs, targets):
                predictions.append(
                    {
                        "boxes": output["boxes"].detach().cpu(),
                        "scores": output["scores"].detach().cpu(),
                        "labels": output["labels"].detach().cpu(),
                    }
                )
                ground_truths.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )
                total_images += 1

    if was_training:
        model.train()
    else:
        model.eval()

    class_ids = sorted(class_id for class_id in label_names.keys() if class_id != 0)
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    precision_tp = 0
    precision_fp = 0
    precision_fn = 0
    matched_ious = []
    ap50_values = []
    map_values = []

    for class_id in class_ids:
        class_stats_50 = evaluate_class_at_iou(
            predictions,
            ground_truths,
            class_id,
            iou_threshold=0.5,
            score_threshold=score_threshold,
        )
        precision_tp += class_stats_50["tp"]
        precision_fp += class_stats_50["fp"]
        precision_fn += class_stats_50["fn"]
        matched_ious.append(class_stats_50["mean_iou"])
        ap50_values.append(class_stats_50["ap"])

        class_ap_values = []
        for iou_threshold in iou_thresholds:
            stats = evaluate_class_at_iou(
                predictions,
                ground_truths,
                class_id,
                iou_threshold=iou_threshold,
                score_threshold=None,
            )
            class_ap_values.append(stats["ap"])

        map_values.append(sum(class_ap_values) / len(class_ap_values) if class_ap_values else 0.0)

    precision = precision_tp / max(precision_tp + precision_fp, 1)
    recall = precision_tp / max(precision_tp + precision_fn, 1)
    mean_iou = sum(matched_ious) / max(len(matched_ious), 1)
    map50 = sum(ap50_values) / max(len(ap50_values), 1)
    map50_95 = sum(map_values) / max(len(map_values), 1)
    avg_loss = total_loss / max(total_images, 1)

    metrics = {
        "loss": avg_loss,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "num_images": total_images,
    }

    print(
        f"Test loss={metrics['loss']:.4f} | IoU={metrics['mean_iou']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
        f"mAP@0.5={metrics['map50']:.4f} | mAP@0.5:0.95={metrics['map50_95']:.4f}"
    )

    return metrics


def test_model(
    model,
    test_loader,
    device,
    output_dir="predictions",
    score_threshold=0.5,
    label_names=None,
    max_images=12,
):
    model.to(device)
    model.eval()

    if label_names is None:
        label_names = {1: "fire", 2: "smoke"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(test_loader, start=1):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for image_idx, (image, output) in enumerate(zip(images, outputs)):
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()

                keep = scores > score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                image_uint8 = (image.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
                label_texts = [
                    f"{label_names.get(int(label), str(int(label)))}: {float(score):.2f}"
                    for label, score in zip(labels, scores)
                ]

                if len(boxes) > 0:
                    drawn = draw_bounding_boxes(
                        image_uint8,
                        boxes,
                        labels=label_texts,
                        colors="red",
                        width=3,
                        font_size=16,
                    )
                else:
                    drawn = image_uint8

                out_path = output_dir / f"pred_{step:04d}_{image_idx:02d}.png"
                to_pil_image(drawn).save(out_path)
                print(f"saved {out_path} | detections={len(boxes)}")
                saved_images += 1

                if saved_images >= max_images:
                    return
            

def main() -> None:
    transforms = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True)
    ])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root = "data/detection/fire_smoke_coco"
    dataset = CocoDetectionDataset(
        images_root=os.path.join(root, "train"),
        annotation_path=os.path.join(root, "annotations", "instances_train.json"),
        transforms=transforms    
    )

    test_dataset = CocoDetectionDataset(
        images_root=os.path.join(root, "test"),
        annotation_path=os.path.join(root, "annotations", "instances_test.json"),
        transforms=transforms    
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 0 = background, 1 = fire, 2 = smoke
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=3)

    model_state_dict = torch.load("fire_smoke.pt", map_location=device)
    model.load_state_dict(model_state_dict)

    print(model)

    model, history = train_model(
        model=model,
        train_loader=loader,
        device=device,
        epochs=10,
        checkpoint_path="checkpoints/fasterrcnn_fire_smoke.pt",
    )

    print(f"Training complete. Final loss: {history[-1]:.4f}")
    torch.save(model.state_dict(), "fire_smoke.pt")

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        score_threshold=0.5,
    )



if __name__ == "__main__":
    main()

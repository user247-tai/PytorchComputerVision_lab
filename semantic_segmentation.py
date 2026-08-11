from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms.v2 import ColorJitter, Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToDtype, ToPureTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_segmentation_masks

from datasets import SemanticSegmentationDataset


NUM_CLASSES = 12
BACKGROUND_INDEX = 0
INPUT_SIZE = (360, 480)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.max() > 1.0:
        images = images / 255.0
    return images


def build_model(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = False):
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        aux_loss=True,
    )


def compute_segmentation_loss(outputs, masks, criterion, aux_weight=0.4):
    loss = criterion(outputs["out"], masks)
    if "aux" in outputs:
        loss = loss + aux_weight * criterion(outputs["aux"], masks)
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    for step, (images, masks) in enumerate(loader, start=1):
        images = normalize_images(images.to(device))
        masks = masks.to(device, dtype=torch.long)

        outputs = model(images)
        loss = compute_segmentation_loss(outputs, masks, criterion)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss encountered at epoch {epoch}, step {step}: {loss.item()}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        # scheduler.step()
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


def update_confusion_matrix(confusion_matrix, preds, targets, num_classes):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    indices = num_classes * targets + preds
    confusion_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def compute_iou_from_confusion(confusion_matrix):
    confusion_matrix = confusion_matrix.to(torch.float32)
    intersection = torch.diag(confusion_matrix)
    ground_truth = confusion_matrix.sum(dim=1)
    predicted = confusion_matrix.sum(dim=0)
    union = ground_truth + predicted - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return iou


def evaluate_model(
    model,
    test_loader,
    device,
    num_classes=NUM_CLASSES,
    background_index=BACKGROUND_INDEX,
):
    model.to(device)
    was_training = model.training
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_pixels = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.inference_mode():
        for images, masks in test_loader:
            images = normalize_images(images.to(device))
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            logits = outputs["out"]
            loss = compute_segmentation_loss(outputs, masks, criterion)

            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            total_correct += int((preds == masks).sum().item())
            total_pixels += int(masks.numel())
            update_confusion_matrix(confusion_matrix, preds.cpu(), masks.cpu(), num_classes)

    if was_training:
        model.train()
    else:
        model.eval()

    avg_loss = total_loss / max(total_samples, 1)
    pixel_accuracy = total_correct / max(total_pixels, 1)
    iou_per_class = compute_iou_from_confusion(confusion_matrix)
    valid_class_mask = torch.ones(num_classes, dtype=torch.bool)
    if 0 <= background_index < num_classes:
        valid_class_mask[background_index] = False

    foreground_iou = iou_per_class[valid_class_mask]
    mean_iou = float(foreground_iou.mean().item()) if foreground_iou.numel() > 0 else float(iou_per_class.mean().item())
    class_iou = {class_id: float(iou_per_class[class_id].item()) for class_id in range(num_classes)}

    metrics = {
        "loss": avg_loss,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "class_iou": class_iou,
    }

    class_iou_text = ", ".join(f"c{class_id}={value:.4f}" for class_id, value in class_iou.items())
    print(
        f"Test loss={metrics['loss']:.4f} | pixel_acc={metrics['pixel_accuracy']:.4f} | "
        f"mean_iou={metrics['mean_iou']:.4f} | {class_iou_text}"
    )

    return metrics


def test_model(model, test_loader, device, num_classes=NUM_CLASSES, background_index=BACKGROUND_INDEX):
    return evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        background_index=background_index,
    )


def mask_to_bool_stack(mask: torch.Tensor, ignore_index: int = 0) -> tuple[torch.Tensor, list[int]]:
    class_ids = [int(label) for label in mask.unique().tolist() if int(label) != ignore_index]
    if not class_ids:
        return mask.unsqueeze(0).bool(), []

    bool_masks = torch.stack([(mask == class_id) for class_id in class_ids], dim=0)
    return bool_masks, class_ids


def model_output_to_bool_masks(logits: torch.Tensor, background_index: int = 0) -> tuple[torch.Tensor, list[int]]:
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(logits.shape)}")
        logits = logits[0]  # (C, H, W)

    if logits.ndim != 3:
        raise ValueError(f"Expected shape (C, H, W), got {tuple(logits.shape)}")

    pred = logits.argmax(dim=0)  # (H, W)
    class_ids = [int(i) for i in pred.unique().tolist() if int(i) != background_index]

    if not class_ids:
        return pred.unsqueeze(0).bool(), []

    bool_masks = torch.stack([(pred == class_id) for class_id in class_ids], dim=0)
    return bool_masks, class_ids


def load_image_tensor(image_path: str | Path, transforms: Compose | None = None) -> torch.Tensor:
    image = decode_image(str(image_path))
    if transforms is not None:
        image = transforms(image)
    return image


def load_mask_tensor(mask_path: str | Path) -> torch.Tensor:
    with Image.open(mask_path) as mask:
        return F.pil_to_tensor(mask).squeeze(0).to(torch.int64)


def resize_label_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=size,
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).to(torch.int64)


def load_segmentation_model(model_path: str | Path, device: str, num_classes: int = NUM_CLASSES):
    model_weights = torch.load(model_path, map_location=device)
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes, aux_loss=True)
    model.load_state_dict(model_weights)
    return model.to(device).eval()


def predict_image(
    model,
    image_path: str | Path,
    device: str,
    transforms: Compose,
    background_index: int = BACKGROUND_INDEX,
):
    image = load_image_tensor(image_path, transforms=transforms)
    with torch.inference_mode():
        logits = model(image.to(device).unsqueeze(0))["out"]
    pred_mask, pred_class_ids = model_output_to_bool_masks(logits=logits, background_index=background_index)
    return image, logits, pred_mask, pred_class_ids



def save_colorized_label_map(label_map: torch.Tensor, save_path: str | Path) -> None:
    palette = torch.tensor(
        [
            [240, 240, 240],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [128, 128, 0],
            [0, 128, 128],
        ],
        dtype=torch.uint8,
    )
    label_map = label_map.detach().cpu().to(torch.long)
    label_map = label_map.clamp(min=0, max=palette.shape[0] - 1)
    color_image = palette[label_map].permute(2, 0, 1).contiguous()
    to_pil_image(color_image).save(save_path)


def predict_and_visualize(
    model,
    image_path: str | Path,
    device: str,
    transforms: Compose,
    save_path: str = "test_segmentation_pred.png",
    mask_path: str | Path | None = None,
):
    image, logits, pred_mask, pred_class_ids = predict_image(
        model=model,
        image_path=image_path,
        device=device,
        transforms=transforms,
    )
    visualize_mask_overlay(
        image=image,
        masks=pred_mask,
        class_ids=pred_class_ids,
        save_path=save_path,
    )

    pred_label_map = logits.argmax(dim=1)[0]
    pred_map_path = str(Path(save_path).with_name(Path(save_path).stem + "_labels.png"))
    save_colorized_label_map(pred_label_map, pred_map_path)

    if not pred_class_ids:
        print("Warning: prediction contains only background, so the overlay looks like the input image.")

    if mask_path is not None:
        gt_mask = load_mask_tensor(mask_path)
        gt_mask = resize_label_mask(gt_mask, size=pred_label_map.shape[-2:])
        gt_bool_mask, gt_class_ids = mask_to_bool_stack(mask=gt_mask)
        gt_save_path = str(Path(save_path).with_name(Path(save_path).stem + "_gt.png"))
        visualize_mask_overlay(
            image=image,
            masks=gt_bool_mask,
            class_ids=gt_class_ids,
            save_path=gt_save_path,
        )
    else:
        gt_class_ids = []

    return logits, pred_class_ids, gt_class_ids


def visualize_mask_overlay(image: torch.Tensor, masks: torch.Tensor, class_ids: list[int], save_path: str) -> None:
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)

    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "orange",
        "purple",
        "pink",
        "brown",
        "lime",
    ]
    id_to_color = {id_: colors[id_ - 1] for id_ in class_ids}
    drawn = draw_segmentation_masks(image, masks.cpu(), alpha=0.6, colors=list(id_to_color.values()))
    to_pil_image(drawn).save(save_path)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root_dir = "data/segmentation/semantic/dataset1"

    train_transforms = Compose([
        Resize(INPUT_SIZE),
        # RandomHorizontalFlip(p=0.5),
        ToDtype(torch.float, scale=True),
        ToPureTensor(),
    ])

    valid_transforms = Compose([
        Resize(INPUT_SIZE),
        ToDtype(torch.float, scale=True),
        ToPureTensor(),
    ])

    train_dataset = SemanticSegmentationDataset(
        root_dir=root_dir,
        split="train",
        transforms=train_transforms,
    )

    test_dataset = SemanticSegmentationDataset(
        root_dir=root_dir,
        split="test",
        transforms=valid_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        # pin_memory=torch.cuda.is_available(),
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        # pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=NUM_CLASSES, pretrained_backbone=False)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=50,
        checkpoint_path="checkpoints/deeplabv3_lane.pt",
    )

    print(f"Training complete. Final loss: {history[-1]:.4f}")
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
        background_index=BACKGROUND_INDEX,
    )
    torch.save(model.state_dict(), "lane_segmentation.pt")

    # model_path = "/home/server/computer_vision_lab/lane_segmentation.pt"
    # image_path = "/home/server/computer_vision_lab/data/segmentation/semantic/dataset1/test/images/0016E5_08111.png"
    # mask_path = "/home/server/computer_vision_lab/data/segmentation/semantic/dataset1/test/masks/0016E5_08111.png"

    # model = load_segmentation_model(model_path=model_path, device=device, num_classes=NUM_CLASSES)
    # logits, pred_class_ids, gt_class_ids = predict_and_visualize(
    #     model=model,
    #     image_path=image_path,
    #     device=device,
    #     transforms=valid_transforms,
    #     save_path="test_segmentation_pred.png",
    #     mask_path=mask_path,
    # )
    # print(f"Saved test_segmentation_pred.png | class_ids={pred_class_ids}")
    # print(f"Saved test_segmentation_pred_gt.png | class_ids={gt_class_ids}")
    # print(f"Pred unique labels: {sorted(logits.argmax(dim=1)[0].unique().tolist())}")


if __name__ == "__main__":
    main()

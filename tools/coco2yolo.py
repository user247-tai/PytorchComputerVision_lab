import json
import os
import glob
import shutil
from tqdm import tqdm

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Converts a bounding box from COCO format to YOLO format.

    COCO format : [x_min, y_min, width, height] (Absolute pixel values)
    YOLO format : [x_center, y_center, width, height] (Normalized values [0, 1])

    Args:
        coco_bbox (list): Absolute bounding box [x_min, y_min, width, height].
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        list: YOLO bounding box [x_center, y_center, w_norm, h_norm] rounded to 6 decimal places.
    """
    x_min, y_min, w_pixel, h_pixel = coco_bbox

    # Calculate center coordinates (x_center, y_center)
    x_center = x_min + (w_pixel / 2.0)
    y_center = y_min + (h_pixel / 2.0)

    # Normalize coordinates relative to image width and height
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w_pixel / img_width
    h_norm = h_pixel / img_height

    # Clamp values to range [0.0, 1.0] to avoid potential rounding edge cases
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return [
        round(x_center_norm, 6),
        round(y_center_norm, 6),
        round(w_norm, 6),
        round(h_norm, 6)
    ]


def convert_coco_split_to_yolo(coco_json_path, coco_img_dir, output_split_dir):
    """
    Converts a COCO JSON split into YOLO format text label files and copies corresponding images.

    Args:
        coco_json_path (str): Path to the COCO JSON annotation file.
        coco_img_dir (str): Directory containing the images for this split.
        output_split_dir (str): Output directory path for this split (e.g., 'yolo_output/train').

    Returns:
        list: Sorted list of class names extracted from the COCO categories.
    """
    if not os.path.exists(coco_json_path):
        print(f"⚠️  COCO annotation file not found: {coco_json_path}. Skipping.")
        return []

    # Load COCO JSON file
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 1. Map category_id to continuous zero-indexed YOLO class IDs
    categories = coco_data.get("categories", [])
    # Sort categories by ID to ensure consistent class ordering
    categories = sorted(categories, key=lambda x: x["id"])
    
    category_map = {}
    class_names = []
    for yolo_class_id, cat in enumerate(categories):
        category_map[cat["id"]] = yolo_class_id
        class_names.append(cat["name"])

    # 2. Map image_id to image metadata dictionary
    images_dict = {img["id"]: img for img in coco_data.get("images", [])}

    # 3. Group annotations by image_id
    annotations_by_img = {}
    for ann in coco_data.get("annotations", []):
        # Ignore crowd regions if specified
        if ann.get("iscrowd", 0) == 1:
            continue
        
        img_id = ann["image_id"]
        annotations_by_img.setdefault(img_id, []).append(ann)

    # Prepare output folders: split/images and split/labels
    out_images_dir = os.path.join(output_split_dir, "images")
    out_labels_dir = os.path.join(output_split_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    print(f"\n🔄 Converting '{os.path.basename(coco_json_path)}' ({len(images_dict)} images)...")

    # 4. Generate YOLO text files and copy image files
    for img_id, img_info in tqdm(images_dict.items()):
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        base_name = os.path.splitext(file_name)[0]
        label_txt_path = os.path.join(out_labels_dir, f"{base_name}.txt")

        # Get annotations for the current image
        img_annotations = annotations_by_img.get(img_id, [])

        yolo_lines = []
        for ann in img_annotations:
            cat_id = ann["category_id"]
            if cat_id not in category_map:
                continue

            yolo_class_id = category_map[cat_id]
            coco_bbox = ann["bbox"]  # [x_min, y_min, width, height]

            # Convert bounding box to YOLO normalized format
            yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_w, img_h)
            
            # Format label string: <class_id> <x_center> <y_center> <width> <height>
            line = f"{yolo_class_id} " + " ".join(map(str, yolo_bbox))
            yolo_lines.append(line)

        # Write converted annotations to text file (create empty file if no annotations)
        with open(label_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # Copy original image to the target YOLO images folder
        src_img_path = os.path.join(coco_img_dir, file_name)
        dst_img_path = os.path.join(out_images_dir, file_name)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

    return class_names


def generate_data_yaml(output_dir, class_names):
    """
    Generates a standard YOLO data.yaml configuration file.

    Args:
        output_dir (str): Target root directory where data.yaml will be saved.
        class_names (list): List of class name strings.
    """
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ Generated dataset config: {yaml_path}")


if __name__ == "__main__":
    # --- CONFIGURATION PATHS ---
    COCO_DATASET_DIR = "fire_smoke_coco"  # Path to your source COCO dataset
    YOLO_OUTPUT_DIR = "fire_smoke_yolo"   # Path where converted YOLO dataset will be saved

    # Define target splits to convert
    SPLITS = ["train", "valid", "test"]

    all_class_names = []

    for split in SPLITS:
        # COCO json path: e.g., fire_smoke_coco/annotations/instances_train.json
        json_path = os.path.join(COCO_DATASET_DIR, "annotations", f"instances_{split}.json")
        # COCO image dir: e.g., fire_smoke_coco/train
        img_dir = os.path.join(COCO_DATASET_DIR, split)
        # Output split dir: e.g., fire_smoke_yolo/train
        output_split_dir = os.path.join(YOLO_OUTPUT_DIR, split)

        class_names = convert_coco_split_to_yolo(
            coco_json_path=json_path,
            coco_img_dir=img_dir,
            output_split_dir=output_split_dir
        )
        
        if class_names and not all_class_names:
            all_class_names = class_names

    # Create the data.yaml file required for YOLO training
    if all_class_names:
        generate_data_yaml(YOLO_OUTPUT_DIR, all_class_names)
    else:
        print("⚠️  No class names extracted. Skipping data.yaml generation.")
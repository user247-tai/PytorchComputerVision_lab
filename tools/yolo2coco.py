import json
import os
import glob
import imagesize
from tqdm import tqdm
import shutil

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Converts a bounding box from YOLO format to COCO format.

    YOLO format : [x_center, y_center, width, height] (Normalized values [0, 1])
    COCO format : [x_min, y_min, width, height] (Absolute pixel values)

    Args:
        yolo_bbox (list): Normalized bounding box [x_center, y_center, w, h].
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        list: COCO bounding box [x_min, y_min, w_pixel, h_pixel] rounded to 2 decimal places.
    """
    x_center, y_center, w, h = yolo_bbox
    
    # Scale normalized width and height back to absolute pixel dimensions
    w_pixel = w * img_width
    h_pixel = h * img_height
    
    # Calculate top-left corner (x_min, y_min) from center coordinates
    x_min = (x_center - w / 2) * img_width
    y_min = (y_center - h / 2) * img_height
    
    # Round float values to avoid excessively long precision issues
    return [round(x_min, 2), round(y_min, 2), round(w_pixel, 2), round(h_pixel, 2)]


def convert_yolo_split_to_coco(yolo_dir, split, categories, output_dir):
    """
    Converts a specific split (train/valid/test) of a YOLO dataset into a COCO JSON annotation file.

    Args:
        yolo_dir (str): Root directory of the YOLO dataset.
        split (str): Split name ('train', 'valid', or 'test').
        categories (list): List of COCO category dictionaries containing class IDs and names.
        output_dir (str): Destination directory where the resulting JSON file will be saved.
    """
    images_dir = os.path.join(yolo_dir, split, "images")
    labels_dir = os.path.join(yolo_dir, split, "labels")
    
    # Skip processing if the image directory for this split does not exist
    if not os.path.exists(images_dir):
        print(f"⚠️  Directory not found: {images_dir}. Skipping split '{split}'.")
        return

    # Initialize the root COCO JSON structure
    coco_data = {
        "info": {
            "description": "Converted from YOLO Dataset",
            "version": "1.0"
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # Support common image extensions
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.BMP", "*.JPG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    # Unique counter IDs required by COCO schema
    image_id = 1
    annotation_id = 1

    print(f"\n🔄 Processing '{split}' split ({len(image_paths)} images)...")

    for img_path in tqdm(image_paths):
        # 1. Fetch image dimensions without loading full pixels into RAM
        try:
            width, height = imagesize.get(img_path)
        except Exception as e:
            print(f"Error reading image metadata for {img_path}: {e}")
            continue

        file_name = os.path.basename(img_path)
        
        # Append image metadata entry to the COCO "images" array
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # 2. Locate corresponding YOLO label (.txt) file
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Parse labels if the text file exists
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # Ignore malformed or non-bbox rows
                
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]

                # Convert YOLO normalized coordinates to COCO absolute pixel format
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)
                area = round(coco_bbox[2] * coco_bbox[3], 2)

                # Append object annotation entry to the COCO "annotations" array
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,  # Keeps class IDs (0: fire, 1: smoke)
                    "bbox": coco_bbox,
                    "area": area,
                    "segmentation": [],       # Left empty for standard Bounding Box datasets
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # 3. Save the constructed dictionary into a JSON file
    os.makedirs(output_dir, exist_ok=True)
    out_json_path = os.path.join(output_dir, f"instances_{split}.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✅ Successfully created: {out_json_path}")


if __name__ == "__main__":
    # --- DATASET PATH CONFIGURATION ---
    YOLO_DATASET_DIR = "/home/server/computer_vision_lab/data/detection/fire_smoke"      # Path to your source YOLO dataset
    OUTPUT_COCO_DIR = "/home/server/computer_vision_lab/data/detection/fire_smoke_coco"   # Output directory for the COCO formatted dataset

    # Class definitions extracted from your data.yaml
    # ['fire', 'smoke'] maps directly to IDs 0 and 1
    CLASSES = ['fire', 'smoke']
    CATEGORIES = [
        {"id": idx, "name": name, "supercategory": "fire_smoke"}
        for idx, name in enumerate(CLASSES)
    ]

    # Target splits to convert
    SPLITS = ["train", "valid", "test"]

    # Convert annotations for each split
    for split in SPLITS:
        convert_yolo_split_to_coco(
            yolo_dir=YOLO_DATASET_DIR,
            split=split,
            categories=CATEGORIES,
            output_dir=os.path.join(OUTPUT_COCO_DIR, "annotations")
        )
        
        # Copy image folders to organize the dataset into standard COCO directory layout
        src_img_dir = os.path.join(YOLO_DATASET_DIR, split, "images")
        dst_img_dir = os.path.join(OUTPUT_COCO_DIR, f"{split}")
        
        if os.path.exists(src_img_dir):
            if os.path.exists(dst_img_dir):
                shutil.rmtree(dst_img_dir)
            shutil.copytree(src_img_dir, dst_img_dir)
            print(f"📁 Copied image folder to: {dst_img_dir}")
import os
import shutil
from random import random, seed

import cv2
import kagglehub


def download_dataset():
    """Download dataset using kagglehub and return the path."""
    # Download latest version
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)
    return path


def prepare_dataset(source_path, target_root, val_ratio=0.2, image_size=224):
    """
    Process the dataset: resize images and split into train/test.

    Args:
        source_path: Path to the downloaded dataset (from kagglehub)
        target_root: Path where the processed dataset will be stored
        val_ratio: Ratio of images to use for validation (test)
        image_size: Target size for images
    """
    classes = ["NORMAL", "PNEUMONIA"]
    seed(1)

    # Clean target directory
    for split in ["train", "test"]:
        for cls in classes:
            dir_path = os.path.join(target_root, split, cls)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

    print(f"Scanning files in {source_path}...")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    count = 0

    for root, dirs, files in os.walk(source_path):
        # Check if current directory is a class directory
        current_folder = os.path.basename(root)
        # The dataset uses uppercase 'NORMAL' and 'PNEUMONIA'
        if current_folder in classes:
            label = current_folder
            print(f"Processing class: {label} in {root}")

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    img_path = os.path.join(root, file)

                    # Read and resize
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Failed to read image: {img_path}")
                            continue
                        img = cv2.resize(img, (image_size, image_size))

                        # Determine split
                        split = "test" if random() < val_ratio else "train"

                        # Save
                        count += 1
                        dest_filename = f"{count}{ext}"
                        dest_path = os.path.join(
                            target_root, split, label, dest_filename
                        )
                        cv2.imwrite(dest_path, img)

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

    print(f"Processed {count} images.")


if __name__ == "__main__":
    dataset_path = download_dataset()
    dataset_home = "src/dataset/chest_xray/"
    prepare_dataset(dataset_path, dataset_home)

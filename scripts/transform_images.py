#!/usr/bin/env python3
"""
Apply torchvision transforms to images listed in train/val/test CSVs
and save results as JPGs in split-specific output folders.

No normalization is applied — InternVL3's processor handles that at inference time.
"""

import csv
import os
from PIL import Image
from torchvision import transforms

DATA_ROOT = "/Users/jackwarren430/Documents/Classes/deep learning/CS-4644-Project/data"
INPUT_DIR = os.path.join(DATA_ROOT, "raw/fgvc-aircraft-2013b/data/images")
LABELS_DIR = os.path.join(DATA_ROOT, "processed/fgvc/labels")
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed/fgvc/images")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

SPLITS = {
    "train": train_transform,
    "val": eval_transform,
    "test": eval_transform,
}


def load_image_ids(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["image_id"] for row in reader]


def main():
    for split, tfm in SPLITS.items():
        csv_path = os.path.join(LABELS_DIR, f"{split}.csv")
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        image_ids = load_image_ids(csv_path)
        print(f"[{split}] Processing {len(image_ids)} images...")

        for i, img_id in enumerate(image_ids, 1):
            img = Image.open(os.path.join(INPUT_DIR, f"{img_id}.jpg")).convert("RGB")
            img = tfm(img)
            img.save(os.path.join(out_dir, f"{img_id}.jpg"), "JPEG")

            if i % 500 == 0 or i == len(image_ids):
                print(f"  [{split}] {i}/{len(image_ids)}")

    print("Done.")


if __name__ == "__main__":
    main()

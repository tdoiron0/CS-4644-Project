#!/usr/bin/env python3
"""
Preprocess images: crop bottom 20px copyright strip and resize to 448x448.
InternVL3 expects 448x448; using native resolution preserves detail for
fine-grained aircraft recognition. No augmentation — that happens in the
Dataset at training time.
"""

import csv
import os
from PIL import Image
from torchvision import transforms

DATA_ROOT = "/Users/seanhall/Desktop/cs4644 project/data"
INPUT_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = DATA_ROOT
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed/fgvc/images")

# Remove bottom 20px copyright strip, then resize to 448x448 (InternVL3 native size)
preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, 0, img.size[0], max(1, img.size[1] - 20)))),
    transforms.Resize((448, 448)),
])

SPLITS = ["train", "val", "test"]


def load_image_ids(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["image_id"] for row in reader]


def main():
    for split in SPLITS:
        csv_path = os.path.join(LABELS_DIR, f"{split}.csv")
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        image_ids = load_image_ids(csv_path)
        print(f"[{split}] Processing {len(image_ids)} images...")

        for i, img_id in enumerate(image_ids, 1):
            img = Image.open(os.path.join(INPUT_DIR, f"{img_id}.jpg")).convert("RGB")
            img = preprocess(img)
            img.save(os.path.join(out_dir, f"{img_id}.jpg"), "JPEG")

            if i % 500 == 0 or i == len(image_ids):
                print(f"  [{split}] {i}/{len(image_ids)}")

    print("Done.")


if __name__ == "__main__":
    main()

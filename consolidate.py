#!/usr/bin/env python3
"""
Consolidate FGVC-Aircraft split files into a single master CSV,
then clean up the individual split/box files.

Produces: labels.csv, taxonomy/ folder
Deletes: all images_*.txt files (splits, bounding boxes, plain lists)
"""

import csv
import os
import shutil

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SPLITS = ["train", "val", "test"]

FILES_TO_DELETE = [
    "images_box.txt",
    "images_train.txt",
    "images_val.txt",
    "images_test.txt",
    "images_variant_train.txt",
    "images_variant_val.txt",
    "images_variant_test.txt",
    "images_variant_trainval.txt",
    "images_family_train.txt",
    "images_family_val.txt",
    "images_family_test.txt",
    "images_family_trainval.txt",
    "images_manufacturer_train.txt",
    "images_manufacturer_val.txt",
    "images_manufacturer_test.txt",
    "images_manufacturer_trainval.txt",
]


def parse_label_file(filename):
    """Parse 'image_id label' lines into a dict {image_id: label}."""
    mapping = {}
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id, label = line.split(" ", 1)
            mapping[image_id] = label
    return mapping


def main():
    manufacturer_map = {}
    family_map = {}
    variant_map = {}

    for split in SPLITS:
        manufacturer_map.update(parse_label_file(f"images_manufacturer_{split}.txt"))
        family_map.update(parse_label_file(f"images_family_{split}.txt"))
        variant_map.update(parse_label_file(f"images_variant_{split}.txt"))

    all_ids = sorted(manufacturer_map.keys())

    assert set(all_ids) == set(family_map.keys()) == set(variant_map.keys()), \
        "Mismatch: not all image IDs appear in all three label types"

    print(f"Total images: {len(all_ids)}")

    output_path = os.path.join(DATA_DIR, "labels.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "manufacturer", "family", "variant"])
        for img_id in all_ids:
            writer.writerow([
                img_id,
                manufacturer_map[img_id],
                family_map[img_id],
                variant_map[img_id],
            ])

    print(f"Written: labels.csv ({len(all_ids)} rows)")

    deleted = 0
    for filename in FILES_TO_DELETE:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
            print(f"Deleted: {filename}")
        else:
            print(f"Skipped (not found): {filename}")

    taxonomy_dir = os.path.join(DATA_DIR, "taxonomy")
    os.makedirs(taxonomy_dir, exist_ok=True)
    for filename in ["families.txt", "manufacturers.txt", "variants.txt"]:
        src = os.path.join(DATA_DIR, filename)
        dst = os.path.join(taxonomy_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved: {filename} -> taxonomy/{filename}")

    print(f"\nDone. Deleted {deleted} files. Moved taxonomy lists to taxonomy/")


if __name__ == "__main__":
    main()

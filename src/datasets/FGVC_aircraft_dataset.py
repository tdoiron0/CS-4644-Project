import torch
from torch.utils.data import Dataset
from PIL import Image
import os 
import csv
import numpy as np
from transformers import AutoProcessor

'''
Expected dataset file structure like 
root_path
    images
        0034309.jpg
        0034958.jpg
        etc.
    labels.csv
'''
class AircraftCaptionDataset(Dataset):
    def __init__(self, csv_path, images_path, processor, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.labels_path = csv_path
        self.images_path = images_path
        
        with open(self.labels_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows: list[dict[str: str]] = list(reader)

    def __len__(self):
        return len(self.label_rows)

    def __getitem__(self, index):
        datapoint = self.label_rows[index]

        img_path = os.path.join(self.images_path, datapoint["image_id"] + ".jpg")
        img_raw = Image.open(img_path).convert("RGB")

        label = f"The manufacturer is {datapoint["manufacturer"]}, the family is {datapoint["family"]}, and the variant is {datapoint["variant"]}."

        return self.processor(
            text="What is the manufacturer, family, and variant of this aircraft?", 
            images=img_raw, 
            return_tensors="pt"
        )
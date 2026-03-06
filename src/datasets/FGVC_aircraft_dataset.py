import torch
from torch.utils.data import Dataset
from PIL import Image
import os 
import csv
import numpy as np
from transformers import AutoProcessor

class AircraftCaptionDataset(Dataset):
    def __init__(self, root_path, processor: AutoProcessor, **kwargs):
        super().__init__(**kwargs)

        self.root_path = root_path
        self.images_path = os.path.join(self.root_path, "images")
        self.labels_path = os.path.join(self.root_path, "labels.csv")
        
        with open(self.labels_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows: list[dict[str: str]] = list(reader)

    def __len__(self):
        return len(self.label_rows)

    def __getitem__(self, index):
        datapoint = self.label_rows[index]

        img_path = os.path.join(self.images_path, datapoint["image_id"] + ".jpg")
        img_raw = Image.open(img_path).convert("RGB")

        label = f"The manufacturer is {datapoint["manufacturer"]}, the family is {datapoint["family"]}, the variant is {datapoint["variant"]}."

        return self.processor(
            text="What is the manufacturer, family, and variant of this aircraft", 
            images=img_raw, 
            return_tensors="pt"
        ), self.processor.tokenizer(label, return_tensors="pt")
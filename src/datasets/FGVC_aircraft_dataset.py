import torch
from torch.utils.data import Dataset
from PIL import Image
import os 
import csv

class AircraftCaptionDataset(Dataset):
    def __init__(self, root_path, **kwargs):
        super().__init__(**kwargs)

        self.root_path = root_path
        self.images_path = os.path.join(root_path, "images")
        self.labels_path = os.path.join(root_path, "labels.csv")
        
        with open(self.labels_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows: list[dict[str: str]] = list(reader)

    def __len__(self):
        return len(self.label_rows)

    def __getitem__(self, index):
        datapoint = self.label_rows[index]
        img_path = os.path.join(self.images_path, datapoint["image_id"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        return img, datapoint["manufacturer"], datapoint["family"], datapoint["variant"]
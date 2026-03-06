import torch
from torch.utils.data import Dataset
from PIL import Image
import os 

class AircraftCaptionDataset(Dataset):
    def __init__(self, root_path, **kwargs):
        super().__init__(**kwargs)

        DIR = os.path.dirname(os.path.abspath(__file__))
        os.path.join(DIR, )

        self.root_path = root_path
        self.images_path = os.path.join(root_path, "images")
        self.labels_path = os.path.join(root_path, "labels.csv")
        self.taxonomy_path = os.path.join(root_path, "taxonomy")
        
        self.families: list[str] = []
        self.manufacturres: list[str] = []
        self.variants: list[str] = []

    def __len__(self):
        pass

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert("RGB")
        return img, 
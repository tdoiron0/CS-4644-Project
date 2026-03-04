import torch
from torch.utils.data import Dataset
from PIL import Image
import os 

class AircraftCaptionDataset(Dataset):
    def __init__(self, image_paths, labels_path, **kwargs):
        super().__init__(**kwargs)

        DIR = os.path.dirname(os.path.abspath(__file__))
        os.path.join(DIR, )

        self.image_paths = image_paths

    def __len__(self):
        pass

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        return img, 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import csv

QUESTION = "What is the manufacturer, family, and variant of this aircraft?"

QUESTIONS = [
    "What is the manufacturer of this aircraft?",
    "What is the family of this aircraft?",
    "What is the variant of this aircraft?",
]

CATEGORY_KEYS = ["manufacturer", "family", "variant"]


class AircraftCaptionDataset(Dataset):
    """
    Original dataset: one combined Q/A per image (kept for backward compat).
    """

    def __init__(self, csv_path, images_path, processor):
        self.processor = processor
        self.images_path = images_path

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows = list(reader)

        dummy_image = Image.new("RGB", (448, 448))
        prompt_only = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": QUESTION},
            ]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.prompt_len = prompt_only["input_ids"].shape[-1]

    def __len__(self):
        return len(self.label_rows)

    def __getitem__(self, index):
        datapoint = self.label_rows[index]

        img_path = os.path.join(self.images_path, datapoint["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        answer = (
            f"The manufacturer is {datapoint['manufacturer']}, "
            f"the family is {datapoint['family']}, "
            f"and the variant is {datapoint['variant']}."
        )

        full = self.processor.apply_chat_template(
            [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QUESTION},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:self.prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"].squeeze(0),
            "pixel_values": full["pixel_values"].squeeze(0),
            "labels": labels,
        }


class AircraftQADataset(Dataset):
    """
    3 separate Q/A pairs per image (manufacturer, family, variant).
    Each answer is only the label string. Supports optional training augmentation.
    """

    def __init__(self, csv_path, images_path, processor, train=False):
        self.processor = processor
        self.images_path = images_path
        self.train = train

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows = list(reader)

        if train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augment = None

        self.prompt_lens = {}
        dummy_image = Image.new("RGB", (448, 448))
        for q_idx, question in enumerate(QUESTIONS):
            prompt_only = self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": question},
                ]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            self.prompt_lens[q_idx] = prompt_only["input_ids"].shape[-1]

    def __len__(self):
        return len(self.label_rows) * 3

    def __getitem__(self, index):
        image_idx = index // 3
        q_idx = index % 3

        datapoint = self.label_rows[image_idx]
        question = QUESTIONS[q_idx]
        category = CATEGORY_KEYS[q_idx]
        answer = datapoint[category]

        img_path = os.path.join(self.images_path, datapoint["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.augment is not None:
            image = self.augment(image)

        full = self.processor.apply_chat_template(
            [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:self.prompt_lens[q_idx]] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"].squeeze(0),
            "pixel_values": full["pixel_values"].squeeze(0),
            "labels": labels,
            "category": category,
        }

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import csv

QUESTION = "What is the manufacturer, family, and variant of this aircraft?"


class AircraftCaptionDataset(Dataset):
    """
    Expected file structure:
        images_path/
            0034309.jpg
            0034958.jpg
            ...
        csv_path  (CSV with columns: image_id, manufacturer, family, variant)
    """

    def __init__(self, csv_path, images_path, processor):
        self.processor = processor
        self.images_path = images_path

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.label_rows = list(reader)

        # The prompt is always the same question, so compute its token length once.
        # Use a dummy image so the processor generates the correct image token count.
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

        # Tokenize the full conversation (prompt + answer)
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

        # Build labels: -100 for prompt tokens, real IDs for answer tokens
        input_ids = full["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:self.prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"].squeeze(0),
            "pixel_values": full["pixel_values"].squeeze(0),
            "labels": labels,
        }

import json
import random
import torch
from torch.utils.data import Dataset


class AircraftTextDataset(Dataset):
    """
    Text-only dataset for domain-adaptive pretraining on aircraft Wikipedia text.

    Loads a JSONL corpus (one JSON object per article), splits articles into
    train/val/test by article, then tokenizes and chunks each article's
    full_clean_text into fixed-length sequences for causal language modeling.
    """

    def __init__(self, jsonl_path, processor, max_length=512, split="train", seed=42):
        self.processor = processor
        self.max_length = max_length

        # Load all articles
        with open(jsonl_path, "r") as f:
            articles = [json.loads(line) for line in f if line.strip()]

        # Deterministic shuffle and 80/10/10 split by article
        rng = random.Random(seed)
        indices = list(range(len(articles)))
        rng.shuffle(indices)

        n = len(articles)
        train_end = int(0.8 * n)
        val_end = train_end + int(0.1 * n)

        if split == "train":
            selected = [articles[i] for i in indices[:train_end]]
        elif split == "val":
            selected = [articles[i] for i in indices[train_end:val_end]]
        elif split == "test":
            selected = [articles[i] for i in indices[val_end:]]
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        # Tokenize all selected articles and chunk into max_length sequences
        # Temporarily raise model_max_length to avoid spurious warnings —
        # we chunk to max_length ourselves.
        orig_max = processor.tokenizer.model_max_length
        processor.tokenizer.model_max_length = int(1e9)

        self.chunks = []
        for article in selected:
            text = article.get("full_clean_text", "")
            if not text:
                continue

            token_ids = processor.tokenizer.encode(text, add_special_tokens=False)

            for start in range(0, len(token_ids), max_length):
                chunk = token_ids[start : start + max_length]
                if len(chunk) < 32:
                    # Skip very short trailing chunks
                    continue
                self.chunks.append(chunk)

        processor.tokenizer.model_max_length = orig_max

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Pad to max_length if needed (for batching)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

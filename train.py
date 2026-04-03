#!/usr/bin/env python3
"""
SFT training script for InternVL3-2B on FGVC-Aircraft.
3 separate Q/A per image (manufacturer, family, variant).
LoRA + projector fine-tuning, MPS-compatible.
"""

import csv
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.models import model_factory
from src.datasets.FGVC_aircraft_dataset import AircraftQADataset, CATEGORY_KEYS
from constants.constants import *

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_EPOCHS = 10
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_NEW_TOKENS = 32
LOG_EVERY = 50

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch(batch, device, dtype):
    return {
        k: v.to(device=device, dtype=dtype) if k == "pixel_values"
        else v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def token_accuracy_for_batch(logits, labels):
    """Token-level accuracy on answer tokens (labels != -100)."""
    preds = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if mask.sum() == 0:
        return 0.0, 0
    correct = (preds[mask] == shifted_labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


def cosine_lr(optimizer, step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── Training ─────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device, dtype, epoch, total_steps_so_far, total_steps):
    model.train()
    running_loss = 0.0
    cat_correct = {k: 0 for k in CATEGORY_KEYS}
    cat_total = {k: 0 for k in CATEGORY_KEYS}
    num_steps = len(loader)
    optimizer.zero_grad()

    warmup_steps = int(total_steps * WARMUP_RATIO)

    for step, batch in enumerate(loader):
        global_step = total_steps_so_far + step
        lr = cosine_lr(optimizer, global_step, total_steps, warmup_steps, LEARNING_RATE)

        categories = batch.pop("category")
        inputs = move_batch(batch, device, dtype)

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += outputs.loss.item()

        correct, total = token_accuracy_for_batch(outputs.logits, inputs["labels"])
        for i, cat in enumerate(categories):
            cat_correct[cat] += correct
            cat_total[cat] += total

        if (step + 1) % LOG_EVERY == 0 or (step + 1) == num_steps:
            avg = running_loss / (step + 1)
            ppl = math.exp(min(avg, 100))
            overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
            print(
                f"  [Train] Epoch {epoch} | Step {step+1}/{num_steps} | "
                f"loss={avg:.4f} | ppl={ppl:.2f} | acc={overall_acc:.4f} | lr={lr:.2e}"
            )

    avg_loss = running_loss / max(1, num_steps)
    avg_ppl = math.exp(min(avg_loss, 100))
    overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
    cat_acc = {k: cat_correct[k] / max(1, cat_total[k]) for k in CATEGORY_KEYS}

    return {
        "train_loss": avg_loss,
        "train_ppl": avg_ppl,
        "train_acc_overall": overall_acc,
        **{f"train_acc_{k}": cat_acc[k] for k in CATEGORY_KEYS},
    }


# ── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, processor, device, dtype):
    model.eval()

    # Loss-based metrics
    running_loss = 0.0
    cat_loss = {k: 0.0 for k in CATEGORY_KEYS}
    cat_loss_count = {k: 0 for k in CATEGORY_KEYS}

    # Generation-based exact match
    cat_correct = {k: 0 for k in CATEGORY_KEYS}
    cat_total = {k: 0 for k in CATEGORY_KEYS}

    num_steps = len(loader)

    for step, batch in enumerate(loader):
        categories = batch.pop("category")
        inputs = move_batch(batch, device, dtype)

        outputs = model(**inputs)
        running_loss += outputs.loss.item()

        for i, cat in enumerate(categories):
            cat_loss[cat] += outputs.loss.item()
            cat_loss_count[cat] += 1

        # Generation-based exact match accuracy
        prompt_ids = inputs["input_ids"].clone()
        labels = inputs["labels"]
        prompt_len = (labels[0] == -100).sum().item()
        gen_input = {
            "input_ids": prompt_ids[:, :prompt_len],
            "attention_mask": inputs["attention_mask"][:, :prompt_len],
            "pixel_values": inputs["pixel_values"],
        }

        output_ids = model.generate(**gen_input, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated_ids = output_ids[0, prompt_len:]
        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        answer_ids = labels[0][labels[0] != -100]
        expected_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        for i, cat in enumerate(categories):
            cat_total[cat] += 1
            if generated_text.lower() == expected_text.lower():
                cat_correct[cat] += 1

        if (step + 1) % LOG_EVERY == 0 or (step + 1) == num_steps:
            overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
            avg = running_loss / (step + 1)
            print(f"  [Val] Step {step+1}/{num_steps} | loss={avg:.4f} | acc={overall_acc:.4f}")

    avg_loss = running_loss / max(1, num_steps)
    avg_ppl = math.exp(min(avg_loss, 100))
    overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
    cat_acc = {k: cat_correct[k] / max(1, cat_total[k]) for k in CATEGORY_KEYS}
    cat_avg_loss = {k: cat_loss[k] / max(1, cat_loss_count[k]) for k in CATEGORY_KEYS}

    return {
        "val_loss": avg_loss,
        "val_ppl": avg_ppl,
        "val_acc_overall": overall_acc,
        **{f"val_acc_{k}": cat_acc[k] for k in CATEGORY_KEYS},
        **{f"val_loss_{k}": cat_avg_loss[k] for k in CATEGORY_KEYS},
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    device = get_device()
    print(f"Using device: {device}")

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    # ── Model ──
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
        train_projector=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    # ── Datasets ──
    train_dataset = AircraftQADataset(
        csv_path=FGVC_TRAIN_LABELS,
        images_path=FGVC_TRAIN_IMAGES,
        processor=processor,
        train=True,
    )
    val_dataset = AircraftQADataset(
        csv_path=FGVC_VAL_LABELS,
        images_path=FGVC_VAL_IMAGES,
        processor=processor,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # MPS can be finicky with workers; set >0 if stable on your setup
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    total_steps = NUM_EPOCHS * len(train_loader)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Steps/epoch: {len(train_loader)} | Total steps: {total_steps}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── CSV logger ──
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_fields = [
        "epoch",
        "train_loss", "train_ppl",
        "train_acc_overall", "train_acc_manufacturer", "train_acc_family", "train_acc_variant",
        "val_loss", "val_ppl",
        "val_loss_manufacturer", "val_loss_family", "val_loss_variant",
        "val_acc_overall", "val_acc_manufacturer", "val_acc_family", "val_acc_variant",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Training loop ──
    best_val_acc = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        steps_so_far = (epoch - 1) * len(train_loader)
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, model.dtype,
            epoch, steps_so_far, total_steps,
        )
        train_time = time.time() - t0

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        t0 = time.time()
        val_metrics = validate(model, val_loader, processor, device, model.dtype)
        val_time = time.time() - t0

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # ── Log ──
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train: loss={train_metrics['train_loss']:.4f}  ppl={train_metrics['train_ppl']:.2f}  acc={train_metrics['train_acc_overall']:.4f}")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}  ppl={val_metrics['val_ppl']:.2f}  acc={val_metrics['val_acc_overall']:.4f}")
        print(f"  Val per-cat acc:  mfr={val_metrics['val_acc_manufacturer']:.4f}  fam={val_metrics['val_acc_family']:.4f}  var={val_metrics['val_acc_variant']:.4f}")
        print(f"  Time: train={train_time:.0f}s  val={val_time:.0f}s")

        # ── Checkpoint ──
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}")
        model.save_pretrained(epoch_ckpt)
        processor.save_pretrained(epoch_ckpt)
        print(f"  Saved checkpoint: {epoch_ckpt}")

        # ── Best model ──
        if val_metrics["val_acc_overall"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_overall"]
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  New best model! val_acc={best_val_acc:.4f}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

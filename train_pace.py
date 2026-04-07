#!/usr/bin/env python3
"""
PACE ICE training script for InternVL3-2B on FGVC-Aircraft.
3 separate Q/A per image (manufacturer, family, variant).
LoRA + projector fine-tuning, CUDA-targeted.

Two-checkpoint system:
  latest/  – recovery checkpoint, overwritten every SAVE_EVERY steps + end of epoch
  best/    – best validation model, overwritten when val_acc improves

Supports --resume to continue from latest/ across chained SLURM jobs.
Handles SIGTERM for graceful shutdown before PACE time limits.
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.models import model_factory
from src.datasets.FGVC_aircraft_dataset import AircraftQADataset, CATEGORY_KEYS
from constants.constants import MODEL_INTERNVL3_2B

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_EPOCHS = 10
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_NEW_TOKENS = 32
LOG_EVERY = 10
SAVE_EVERY = 2000  # save latest checkpoint every N steps (~10 per epoch)

# ── Globals for SIGTERM ──────────────────────────────────────────────────────

_sigterm_received = False


def _sigterm_handler(signum, frame):
    global _sigterm_received
    _sigterm_received = True
    print("\n[SIGTERM] Received — will save checkpoint and exit after current step.")


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
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


# ── Checkpoint helpers ───────────────────────────────────────────────────────


def save_checkpoint(path, model, processor, optimizer, epoch, global_step,
                    best_val_acc, metrics=None):
    """Save model weights, processor, and training state to a directory."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, os.path.join(path, "training_state.pt"))
    print(f"  [Checkpoint] Saved to {path}  (epoch={epoch}, step={global_step})")


def load_checkpoint(path, model, optimizer, device):
    """Load training state from a checkpoint directory. Returns metadata dict."""
    from peft import PeftModel
    state_path = os.path.join(path, "training_state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"No training_state.pt in {path}")

    adapter_config = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        model.load_adapter(path, adapter_name="default")
        print(f"  [Resume] Loaded adapter weights from {path}")

    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"  [Resume] Restored optimizer state")
    print(f"  [Resume] epoch={state['epoch']}  global_step={state['global_step']}  "
          f"best_val_acc={state['best_val_acc']:.4f}")
    return state


# ── Standalone train-metrics persistence (survives checkpoint overwrites) ────


def _train_metrics_path(output_dir, epoch):
    return os.path.join(output_dir, f"train_metrics_epoch_{epoch}.json")


def save_train_metrics(output_dir, epoch, metrics):
    path = _train_metrics_path(output_dir, epoch)
    with open(path, "w") as f:
        json.dump(metrics, f)
    print(f"  [Metrics] Saved train metrics for epoch {epoch} → {path}")


def load_train_metrics(output_dir, epoch):
    path = _train_metrics_path(output_dir, epoch)
    if os.path.isfile(path):
        with open(path, "r") as f:
            metrics = json.load(f)
        print(f"  [Metrics] Loaded train metrics for epoch {epoch} from {path}")
        return metrics
    return None


def delete_train_metrics(output_dir, epoch):
    path = _train_metrics_path(output_dir, epoch)
    if os.path.isfile(path):
        os.remove(path)


# ── Training ─────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device, dtype, epoch,
                    total_steps_so_far, total_steps, start_step=0,
                    save_latest_fn=None, output_dir=None):
    """
    Train for one epoch. Returns (metrics_dict, last_step_in_epoch, interrupted).
    If SIGTERM is received, returns early with interrupted=True.
    """
    model.train()
    running_loss = 0.0
    cat_loss = {k: 0.0 for k in CATEGORY_KEYS}
    cat_loss_count = {k: 0 for k in CATEGORY_KEYS}
    cat_correct = {k: 0 for k in CATEGORY_KEYS}
    cat_total = {k: 0 for k in CATEGORY_KEYS}
    num_steps = len(loader)
    optimizer.zero_grad()

    warmup_steps = int(total_steps * WARMUP_RATIO)

    for step, batch in enumerate(loader):
        if step < start_step:
            continue

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

        batch_loss = outputs.loss.item()
        running_loss += batch_loss

        for i, cat in enumerate(categories):
            cat_loss[cat] += batch_loss
            cat_loss_count[cat] += 1

        correct, total = token_accuracy_for_batch(outputs.logits, inputs["labels"])
        for i, cat in enumerate(categories):
            cat_correct[cat] += correct
            cat_total[cat] += total

        steps_done = step - start_step + 1

        if steps_done % LOG_EVERY == 0 or (step + 1) == num_steps:
            avg = running_loss / steps_done
            ppl = math.exp(min(avg, 100))
            overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
            print(
                f"  [Train] Epoch {epoch} | Step {step+1}/{num_steps} | "
                f"loss={avg:.4f} | ppl={ppl:.2f} | acc={overall_acc:.4f} | lr={lr:.2e}"
            )

        if (step + 1) % SAVE_EVERY == 0 and save_latest_fn is not None:
            save_latest_fn(epoch=epoch, global_step=global_step)

        if _sigterm_received:
            print(f"  [SIGTERM] Stopping at epoch {epoch}, step {step+1}")
            metrics = _build_train_metrics(running_loss, steps_done, cat_loss, cat_loss_count,
                                           cat_correct, cat_total)
            if output_dir is not None:
                save_train_metrics(output_dir, epoch, metrics)
            if save_latest_fn is not None:
                save_latest_fn(epoch=epoch, global_step=global_step, metrics=metrics)
            return metrics, step, True

    steps_done = num_steps - start_step
    return _build_train_metrics(running_loss, steps_done, cat_loss, cat_loss_count,
                                cat_correct, cat_total), num_steps - 1, False


def _build_train_metrics(running_loss, steps_done, cat_loss, cat_loss_count,
                         cat_correct, cat_total):
    avg_loss = running_loss / max(1, steps_done)
    avg_ppl = math.exp(min(avg_loss, 100))
    overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
    cat_acc = {k: cat_correct[k] / max(1, cat_total[k]) for k in CATEGORY_KEYS}
    cat_avg_loss = {k: cat_loss[k] / max(1, cat_loss_count[k]) for k in CATEGORY_KEYS}
    cat_ppl = {k: math.exp(min(cat_avg_loss[k], 100)) for k in CATEGORY_KEYS}

    return {
        "train_loss": avg_loss,
        "train_ppl": avg_ppl,
        "train_acc_overall": overall_acc,
        **{f"train_loss_{k}": cat_avg_loss[k] for k in CATEGORY_KEYS},
        **{f"train_ppl_{k}": cat_ppl[k] for k in CATEGORY_KEYS},
        **{f"train_acc_{k}": cat_acc[k] for k in CATEGORY_KEYS},
    }


# ── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, processor, device, dtype):
    model.eval()

    running_loss = 0.0
    cat_loss = {k: 0.0 for k in CATEGORY_KEYS}
    cat_loss_count = {k: 0 for k in CATEGORY_KEYS}
    cat_correct = {k: 0 for k in CATEGORY_KEYS}
    cat_total = {k: 0 for k in CATEGORY_KEYS}

    num_steps = len(loader)

    for step, batch in enumerate(loader):
        categories = batch.pop("category")
        inputs = move_batch(batch, device, dtype)

        outputs = model(**inputs)
        batch_loss = outputs.loss.item()
        running_loss += batch_loss

        for i, cat in enumerate(categories):
            cat_loss[cat] += batch_loss
            cat_loss_count[cat] += 1

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
    cat_ppl = {k: math.exp(min(cat_avg_loss[k], 100)) for k in CATEGORY_KEYS}

    return {
        "val_loss": avg_loss,
        "val_ppl": avg_ppl,
        "val_acc_overall": overall_acc,
        **{f"val_loss_{k}": cat_avg_loss[k] for k in CATEGORY_KEYS},
        **{f"val_ppl_{k}": cat_ppl[k] for k in CATEGORY_KEYS},
        **{f"val_acc_{k}": cat_acc[k] for k in CATEGORY_KEYS},
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="PACE ICE training for InternVL3-2B FGVC-Aircraft")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing processed/ and {train,val,test}.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints and metrics (use scratch on PACE)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to latest/ checkpoint directory to resume from. "
                             "If the path doesn't exist yet, starts fresh.")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to a model_weights.pt state dict to load before training. "
                             "Use this to initialize from a wiki-finetuned checkpoint. "
                             "Ignored if --resume finds a valid checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    # ── Paths ──
    train_images = os.path.join(args.data_root, "processed/fgvc/images/train")
    train_labels = os.path.join(args.data_root, "train.csv")
    val_images = os.path.join(args.data_root, "processed/fgvc/images/val")
    val_labels = os.path.join(args.data_root, "val.csv")

    latest_dir = os.path.join(args.output_dir, "latest")
    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
        train_projector=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    # ── Datasets ──
    train_dataset = AircraftQADataset(
        csv_path=train_labels,
        images_path=train_images,
        processor=processor,
        train=True,
    )
    val_dataset = AircraftQADataset(
        csv_path=val_labels,
        images_path=val_images,
        processor=processor,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Pretrained weights (wiki checkpoint) ──
    if args.pretrained and not (args.resume and os.path.isdir(args.resume)):
        print(f"\n[Pretrained] Loading weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [Pretrained] Loaded model weights successfully")

    # ── Resume ──
    start_epoch = 1
    start_step = 0
    best_val_acc = -1.0

    resumed_train_metrics = None  # train metrics from checkpoint, for missed validation
    if args.resume and os.path.isdir(args.resume):
        print(f"\n[Resume] Loading from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer, device)
        best_val_acc = state["best_val_acc"]
        resumed_train_metrics = state.get("metrics")  # saved at end of training, before val

        resumed_epoch = state["epoch"]
        resumed_step = state["global_step"]
        completed_epoch_step = (resumed_epoch - 1) * steps_per_epoch + steps_per_epoch - 1
        epoch_start_step = (resumed_epoch - 1) * steps_per_epoch

        if resumed_step >= completed_epoch_step:
            start_epoch = resumed_epoch + 1
            start_step = 0
            print(f"  [Resume] Epoch {resumed_epoch} was complete — starting epoch {start_epoch}")
        else:
            start_epoch = resumed_epoch
            start_step = resumed_step - epoch_start_step + 1
            print(f"  [Resume] Resuming epoch {start_epoch} from step {start_step}/{steps_per_epoch}")
    elif args.resume:
        print(f"[Resume] Path {args.resume} not found — starting fresh")

    # ── CSV logger ──
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_fields = [
        "epoch",
        "train_loss", "train_ppl", "train_acc_overall",
        "train_loss_manufacturer", "train_ppl_manufacturer", "train_acc_manufacturer",
        "train_loss_family", "train_ppl_family", "train_acc_family",
        "train_loss_variant", "train_ppl_variant", "train_acc_variant",
        "val_loss", "val_ppl", "val_acc_overall",
        "val_loss_manufacturer", "val_ppl_manufacturer", "val_acc_manufacturer",
        "val_loss_family", "val_ppl_family", "val_acc_family",
        "val_loss_variant", "val_ppl_variant", "val_acc_variant",
    ]
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Checkpoint save helper (closure over shared state) ──
    def save_latest(epoch, global_step, metrics=None):
        save_checkpoint(latest_dir, model, processor, optimizer,
                        epoch, global_step, best_val_acc, metrics)

    # ── Check for missed validation on resume ──
    def epochs_with_val():
        """Return set of epoch numbers that already have validation in metrics.csv."""
        completed = set()
        if os.path.isfile(csv_path):
            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("val_loss"):
                        completed.add(int(row["epoch"]))
        return completed

    if args.resume and os.path.isdir(args.resume):
        validated_epochs = epochs_with_val()
        last_completed_epoch = start_epoch - 1
        if last_completed_epoch >= 1 and last_completed_epoch not in validated_epochs:
            print(f"\n[Resume] Running missed validation for epoch {last_completed_epoch}...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_metrics = validate(model, val_loader, processor, device, model.dtype)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_metrics = load_train_metrics(args.output_dir, last_completed_epoch)
            if train_metrics is None:
                train_metrics = resumed_train_metrics if resumed_train_metrics else {}
            if not train_metrics:
                print(f"  [WARNING] No train metrics found for epoch {last_completed_epoch} — row will have val only")
            row = {"epoch": last_completed_epoch, **train_metrics, **val_metrics}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
                f.flush()
            delete_train_metrics(args.output_dir, last_completed_epoch)

            print(f"  [Resume] Val epoch {last_completed_epoch}: "
                  f"loss={val_metrics['val_loss']:.4f}  acc={val_metrics['val_acc_overall']:.4f}")

            if val_metrics["val_acc_overall"] > best_val_acc:
                best_val_acc = val_metrics["val_acc_overall"]
                global_step = (last_completed_epoch - 1) * steps_per_epoch + steps_per_epoch - 1
                save_checkpoint(best_dir, model, processor, optimizer,
                                last_completed_epoch, global_step, best_val_acc, val_metrics)
                print(f"  New best model! val_acc={best_val_acc:.4f}")

    # ── Training loop ──
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        steps_so_far = (epoch - 1) * steps_per_epoch
        epoch_start = start_step if epoch == start_epoch else 0

        t0 = time.time()
        train_metrics, last_step, interrupted = train_one_epoch(
            model, train_loader, optimizer, device, model.dtype,
            epoch, steps_so_far, total_steps,
            start_step=epoch_start,
            save_latest_fn=save_latest,
            output_dir=args.output_dir,
        )
        train_time = time.time() - t0

        if interrupted:
            row = {"epoch": epoch, **train_metrics}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
                f.flush()
            print(f"\n[SIGTERM] Exiting after saving checkpoint. Train metrics written to CSV.")
            sys.exit(0)

        # Persist train metrics to standalone JSON (survives checkpoint overwrites + job kills)
        save_train_metrics(args.output_dir, epoch, train_metrics)

        # Save latest before validation (so it's not lost if val crashes)
        global_step = steps_so_far + steps_per_epoch - 1
        save_latest(epoch=epoch, global_step=global_step, metrics=train_metrics)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        val_metrics = validate(model, val_loader, processor, device, model.dtype)
        val_time = time.time() - t0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Log ──
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
            f.flush()
        delete_train_metrics(args.output_dir, epoch)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train: loss={train_metrics['train_loss']:.4f}  "
              f"ppl={train_metrics['train_ppl']:.2f}  "
              f"acc={train_metrics['train_acc_overall']:.4f}")
        print(f"    mfr:  loss={train_metrics['train_loss_manufacturer']:.4f}  "
              f"ppl={train_metrics['train_ppl_manufacturer']:.2f}  "
              f"acc={train_metrics['train_acc_manufacturer']:.4f}")
        print(f"    fam:  loss={train_metrics['train_loss_family']:.4f}  "
              f"ppl={train_metrics['train_ppl_family']:.2f}  "
              f"acc={train_metrics['train_acc_family']:.4f}")
        print(f"    var:  loss={train_metrics['train_loss_variant']:.4f}  "
              f"ppl={train_metrics['train_ppl_variant']:.2f}  "
              f"acc={train_metrics['train_acc_variant']:.4f}")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}  "
              f"ppl={val_metrics['val_ppl']:.2f}  "
              f"acc={val_metrics['val_acc_overall']:.4f}")
        print(f"    mfr:  loss={val_metrics['val_loss_manufacturer']:.4f}  "
              f"ppl={val_metrics['val_ppl_manufacturer']:.2f}  "
              f"acc={val_metrics['val_acc_manufacturer']:.4f}")
        print(f"    fam:  loss={val_metrics['val_loss_family']:.4f}  "
              f"ppl={val_metrics['val_ppl_family']:.2f}  "
              f"acc={val_metrics['val_acc_family']:.4f}")
        print(f"    var:  loss={val_metrics['val_loss_variant']:.4f}  "
              f"ppl={val_metrics['val_ppl_variant']:.2f}  "
              f"acc={val_metrics['val_acc_variant']:.4f}")
        print(f"  Time: train={train_time:.0f}s  val={val_time:.0f}s")

        # ── Best model ──
        if val_metrics["val_acc_overall"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_overall"]
            all_metrics = {**train_metrics, **val_metrics}
            save_checkpoint(best_dir, model, processor, optimizer,
                            epoch, global_step, best_val_acc, all_metrics)
            print(f"  New best model! val_acc={best_val_acc:.4f}")

        start_step = 0

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output_dir}")
    with open(os.path.join(args.output_dir, "TRAINING_COMPLETE"), "w") as f:
        f.write(f"best_val_acc={best_val_acc:.4f}\n")


if __name__ == "__main__":
    main()

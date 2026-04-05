#!/usr/bin/env python3
"""
PACE ICE wiki fine-tuning script for InternVL3-2B on aircraft Wikipedia corpus.
Text-only language modeling (next-token prediction).

Two-checkpoint system:
  latest/  – recovery checkpoint, overwritten every SAVE_EVERY steps + end of epoch
  best/    – best validation model, overwritten when val_loss improves

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

import torch
from torch.utils.data import DataLoader

from src.models import model_factory
from src.datasets.aircraft_text_dataset import AircraftTextDataset
from constants.constants import WIKI_CORPUS_EXPANDED

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_EPOCHS = 10
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LOG_EVERY = 50
SAVE_EVERY = 500  # save latest checkpoint every N steps

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


def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss, metrics=None):
    """Save model weights and training state to a directory."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pt"))
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, os.path.join(path, "training_state.pt"))
    print(f"  [Checkpoint] Saved to {path}  (epoch={epoch}, step={global_step})")


def load_checkpoint(path, model, optimizer, device):
    """Load training state from a checkpoint directory. Returns metadata dict."""
    state_path = os.path.join(path, "training_state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"No training_state.pt in {path}")

    weights_path = os.path.join(path, "model_weights.pt")
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"  [Resume] Loaded model weights from {path}")

    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"  [Resume] Restored optimizer state")
    print(f"  [Resume] epoch={state['epoch']}  global_step={state['global_step']}  "
          f"best_val_loss={state['best_val_loss']:.4f}")
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


def train_one_epoch(model, loader, optimizer, device, epoch,
                    total_steps_so_far, total_steps, start_step=0,
                    save_latest_fn=None, output_dir=None):
    """Train for one epoch. Returns (metrics_dict, last_step_in_epoch, interrupted)."""
    model.train()
    running_loss = 0.0
    steps_done = 0
    num_steps = len(loader)
    optimizer.zero_grad()

    warmup_steps = int(total_steps * WARMUP_RATIO)

    for step, batch in enumerate(loader):
        if step < start_step:
            continue

        global_step = total_steps_so_far + step
        lr = cosine_lr(optimizer, global_step, total_steps, warmup_steps, LEARNING_RATE)
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += outputs.loss.item()
        steps_done += 1

        if steps_done % LOG_EVERY == 0 or (step + 1) == num_steps:
            avg = running_loss / steps_done
            ppl = math.exp(min(avg, 100))
            print(
                f"  [Train] Epoch {epoch} | Step {step+1}/{num_steps} | "
                f"loss={avg:.4f} | ppl={ppl:.2f} | lr={lr:.2e}"
            )

        if (step + 1) % SAVE_EVERY == 0 and save_latest_fn is not None:
            save_latest_fn(epoch=epoch, global_step=global_step)

        if _sigterm_received:
            print(f"  [SIGTERM] Stopping at epoch {epoch}, step {step+1}")
            metrics = _build_train_metrics(running_loss, steps_done)
            if output_dir is not None:
                save_train_metrics(output_dir, epoch, metrics)
            if save_latest_fn is not None:
                save_latest_fn(epoch=epoch, global_step=global_step, metrics=metrics)
            return metrics, step, True

    metrics = _build_train_metrics(running_loss, steps_done)
    return metrics, num_steps - 1, False


def _build_train_metrics(running_loss, steps_done):
    avg_loss = running_loss / max(1, steps_done)
    return {
        "train_loss": avg_loss,
        "train_ppl": math.exp(min(avg_loss, 100)),
    }


# ── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_steps = len(loader)

    for step, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        running_loss += outputs.loss.item()

        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        shift_labels = inputs["labels"][:, 1:]
        mask = shift_labels != -100
        correct += (preds[mask] == shift_labels[mask]).sum().item()
        total += mask.sum().item()

        if (step + 1) % 50 == 0 or (step + 1) == num_steps:
            avg = running_loss / (step + 1)
            acc = correct / total if total > 0 else 0.0
            print(f"  [Val] Step {step+1}/{num_steps} | loss={avg:.4f} | acc={acc:.4f}")

    avg_loss = running_loss / max(1, num_steps)
    acc = correct / total if total > 0 else 0.0
    return {
        "val_loss": avg_loss,
        "val_ppl": math.exp(min(avg_loss, 100)),
        "val_acc": acc,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_metrics(csv_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_loss, val_loss, val_acc, train_ppl, val_ppl = [], [], [], [], [], []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("val_loss"):
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_acc.append(float(row["val_acc"]))
                train_ppl.append(float(row["train_ppl"]))
                val_ppl.append(float(row["val_ppl"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Wiki Finetune — Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_acc, label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Wiki Finetune — Val Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, train_ppl, label="Train")
    axes[2].plot(epochs, val_ppl, label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Wiki Finetune — Perplexity")
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "wiki_metrics.png")
    plt.savefig(out, dpi=150)
    print(f"  [Plot] Saved metrics plot to {out}")
    plt.close()


# ── Args ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="PACE ICE wiki fine-tuning for InternVL3-2B")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints and metrics (use scratch on PACE)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to latest/ checkpoint directory to resume from. "
                             "If the path doesn't exist yet, starts fresh.")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Output dir: {args.output_dir}")

    latest_dir = os.path.join(args.output_dir, "latest")
    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    # ── Datasets ──
    train_dataset = AircraftTextDataset(
        jsonl_path=WIKI_CORPUS_EXPANDED, processor=processor, split="train",
    )
    val_dataset = AircraftTextDataset(
        jsonl_path=WIKI_CORPUS_EXPANDED, processor=processor, split="val",
    )

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Resume ──
    start_epoch = 1
    start_step = 0
    best_val_loss = float("inf")

    resumed_train_metrics = None
    if args.resume and os.path.isdir(args.resume):
        print(f"\n[Resume] Loading from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer, device)
        best_val_loss = state["best_val_loss"]
        resumed_train_metrics = state.get("metrics")

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
    csv_fields = ["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "val_acc"]
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Checkpoint save helper (closure over shared state) ──
    def save_latest(epoch, global_step, metrics=None):
        save_checkpoint(latest_dir, model, optimizer, epoch, global_step, best_val_loss, metrics)

    # ── Missed validation recovery on resume ──
    def epochs_with_val():
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
            val_metrics = validate(model, val_loader, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_metrics = load_train_metrics(args.output_dir, last_completed_epoch)
            if train_metrics is None:
                train_metrics = resumed_train_metrics if resumed_train_metrics else {}
            if not train_metrics:
                print(f"  [WARNING] No train metrics for epoch {last_completed_epoch} — row will have val only")
            row = {"epoch": last_completed_epoch, **train_metrics, **val_metrics}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
                f.flush()
            delete_train_metrics(args.output_dir, last_completed_epoch)

            print(f"  [Resume] Val epoch {last_completed_epoch}: "
                  f"loss={val_metrics['val_loss']:.4f}  acc={val_metrics['val_acc']:.4f}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                global_step = (last_completed_epoch - 1) * steps_per_epoch + steps_per_epoch - 1
                save_checkpoint(best_dir, model, optimizer,
                                last_completed_epoch, global_step, best_val_loss, val_metrics)
                print(f"  New best model! val_loss={best_val_loss:.4f}")

    # ── Training loop ──
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        steps_so_far = (epoch - 1) * steps_per_epoch
        epoch_start = start_step if epoch == start_epoch else 0

        t0 = time.time()
        train_metrics, last_step, interrupted = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            steps_so_far, total_steps,
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

        # Persist train metrics to JSON before validation (survives checkpoint overwrites + job kills)
        save_train_metrics(args.output_dir, epoch, train_metrics)

        # Save latest before validation (so it's not lost if val crashes or job is killed)
        global_step = steps_so_far + steps_per_epoch - 1
        save_latest(epoch=epoch, global_step=global_step, metrics=train_metrics)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        val_metrics = validate(model, val_loader, device)
        val_time = time.time() - t0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
            f.flush()
        delete_train_metrics(args.output_dir, epoch)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train: loss={train_metrics['train_loss']:.4f}  ppl={train_metrics['train_ppl']:.2f}")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}  ppl={val_metrics['val_ppl']:.2f}  "
              f"acc={val_metrics['val_acc']:.4f}")
        print(f"  Time: train={train_time:.0f}s  val={val_time:.0f}s")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(best_dir, model, optimizer,
                            epoch, global_step, best_val_loss, {**train_metrics, **val_metrics})
            print(f"  New best model! val_loss={best_val_loss:.4f}")

        start_step = 0

    plot_metrics(csv_path, args.output_dir)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")
    with open(os.path.join(args.output_dir, "WIKI_TRAINING_COMPLETE"), "w") as f:
        f.write(f"best_val_loss={best_val_loss:.4f}\n")


if __name__ == "__main__":
    main()

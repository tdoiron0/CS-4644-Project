#!/usr/bin/env python3
"""
PACE ICE test script for InternVL3-2B on FGVC-Aircraft test set.

Evaluates two models on the held-out test set, using the same Q/A
format as training (3 separate Q/A per image: manufacturer, family,
variant) with no augmentation:

  1. Baseline:  out-of-the-box InternVL3-2B (state at training step 0).
                Built via the same factory call training uses; no adapter
                is loaded. PEFT initializes LoRA-B to zero, so the
                adapter is a no-op and the model is identical to the
                pretrained InternVL3-2B-hf.
  2. Finetuned: best validation-accuracy LoRA checkpoint from
                training (loaded via model.load_adapter).

For each model we record:
  - test_loss            (cross-entropy on answer tokens)
  - test_ppl             (perplexity = exp(loss))
  - test_acc_overall     (exact-match accuracy of greedy generation)
  - per-category loss / ppl / accuracy for manufacturer / family / variant

Per-sample predictions are streamed to a CSV so partial results survive
a crash or time-out.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from src.models import model_factory
from src.datasets.FGVC_aircraft_dataset import AircraftQADataset, CATEGORY_KEYS

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE = 1            # matches training; AircraftQADataset returns variable-length seqs
MAX_NEW_TOKENS = 32       # matches training validate()
LOG_EVERY = 100           # progress prints


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


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, processor, device, dtype, model_name,
             predictions_csv_path):
    """
    Run full evaluation on `loader`. Mirrors train_pace.validate() exactly:
      - forward pass for loss/perplexity
      - greedy generate() for exact-match accuracy
      - per-category bookkeeping (manufacturer / family / variant)

    Per-sample predictions are streamed to `predictions_csv_path` and
    flushed every 50 steps so they survive an interruption.
    """
    model.eval()

    running_loss = 0.0
    cat_loss = {k: 0.0 for k in CATEGORY_KEYS}
    cat_loss_count = {k: 0 for k in CATEGORY_KEYS}
    cat_correct = {k: 0 for k in CATEGORY_KEYS}
    cat_total = {k: 0 for k in CATEGORY_KEYS}

    num_steps = len(loader)
    n_processed = 0
    n_failed = 0
    t_start = time.time()

    pred_fields = ["index", "category", "expected", "generated", "correct"]
    pred_file = open(predictions_csv_path, "w", newline="")
    pred_writer = csv.DictWriter(pred_file, fieldnames=pred_fields)
    pred_writer.writeheader()
    pred_file.flush()

    try:
        for step, batch in enumerate(loader):
            try:
                categories = batch.pop("category")
                inputs = move_batch(batch, device, dtype)

                outputs = model(**inputs)
                batch_loss = outputs.loss.item()
                running_loss += batch_loss

                for cat in categories:
                    cat_loss[cat] += batch_loss
                    cat_loss_count[cat] += 1

                # Build prompt-only input for generation (mask out the answer).
                labels = inputs["labels"]
                prompt_len = (labels[0] == -100).sum().item()
                gen_input = {
                    "input_ids": inputs["input_ids"][:, :prompt_len],
                    "attention_mask": inputs["attention_mask"][:, :prompt_len],
                    "pixel_values": inputs["pixel_values"],
                }

                output_ids = model.generate(
                    **gen_input,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                generated_ids = output_ids[0, prompt_len:]
                generated_text = processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                answer_ids = labels[0][labels[0] != -100]
                expected_text = processor.tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                ).strip()

                cat = categories[0]
                cat_total[cat] += 1
                is_correct = generated_text.lower() == expected_text.lower()
                if is_correct:
                    cat_correct[cat] += 1

                pred_writer.writerow({
                    "index": step,
                    "category": cat,
                    "expected": expected_text,
                    "generated": generated_text,
                    "correct": int(is_correct),
                })
                n_processed += 1

                if (step + 1) % 50 == 0:
                    pred_file.flush()

                if (step + 1) % LOG_EVERY == 0 or (step + 1) == num_steps:
                    elapsed = time.time() - t_start
                    rate = (step + 1) / max(1e-6, elapsed)
                    eta_min = (num_steps - step - 1) / max(1e-6, rate) / 60.0
                    overall_acc = sum(cat_correct.values()) / max(1, sum(cat_total.values()))
                    avg = running_loss / max(1, n_processed)
                    print(
                        f"  [{model_name}] Step {step+1}/{num_steps} | "
                        f"loss={avg:.4f} | acc={overall_acc:.4f} | "
                        f"{rate:.1f} steps/s | ETA {eta_min:.1f}min",
                        flush=True,
                    )

            except Exception as e:
                n_failed += 1
                print(f"  [WARN] {model_name} step {step} failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                # Best-effort: skip this sample, keep going.
                continue
    finally:
        pred_file.flush()
        pred_file.close()

    avg_loss = running_loss / max(1, n_processed)
    avg_ppl = math.exp(min(avg_loss, 100))
    overall_correct = sum(cat_correct.values())
    overall_total = sum(cat_total.values())
    overall_acc = overall_correct / max(1, overall_total)
    cat_acc = {k: cat_correct[k] / max(1, cat_total[k]) for k in CATEGORY_KEYS}
    cat_avg_loss = {k: cat_loss[k] / max(1, cat_loss_count[k]) for k in CATEGORY_KEYS}
    cat_ppl = {k: math.exp(min(cat_avg_loss[k], 100)) for k in CATEGORY_KEYS}

    return {
        "model": model_name,
        "num_examples_seen": n_processed,
        "num_examples_failed": n_failed,
        "test_loss": avg_loss,
        "test_ppl": avg_ppl,
        "test_acc_overall": overall_acc,
        "test_correct_overall": overall_correct,
        "test_total_overall": overall_total,
        **{f"test_loss_{k}": cat_avg_loss[k] for k in CATEGORY_KEYS},
        **{f"test_ppl_{k}": cat_ppl[k] for k in CATEGORY_KEYS},
        **{f"test_acc_{k}": cat_acc[k] for k in CATEGORY_KEYS},
        **{f"test_correct_{k}": cat_correct[k] for k in CATEGORY_KEYS},
        **{f"test_total_{k}": cat_total[k] for k in CATEGORY_KEYS},
    }


# ── Model builders ───────────────────────────────────────────────────────────


def build_baseline_model(device):
    """
    Build the out-of-the-box model. Same factory call training uses, no
    adapter loaded. PEFT initializes LoRA-B to zero so the freshly-wrapped
    PEFT model produces identical outputs to the pretrained base model.
    """
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
        train_projector=True,
    )
    model.to(device)
    return model, processor


def build_finetuned_model(device, best_dir):
    """Build the model and load the best LoRA adapter on top."""
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
        train_projector=True,
    )
    if not os.path.isfile(os.path.join(best_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"No adapter_config.json in {best_dir} — can't load finetuned weights"
        )
    model.load_adapter(best_dir, adapter_name="default")
    print(f"  [Finetuned] Loaded LoRA adapter from {best_dir}")
    model.to(device)
    return model, processor


# ── Run-one-model wrapper ────────────────────────────────────────────────────


def run_model(which, device, args):
    """
    Build the requested model, run evaluation, save metrics + per-sample CSV.
    Returns the metrics dict (or None on failure).
    """
    print("\n" + "=" * 60)
    if which == "baseline":
        print("BASELINE — out-of-the-box InternVL3-2B (no LoRA training)")
    else:
        print(f"FINETUNED — best LoRA checkpoint from {args.best_dir}")
    print("=" * 60, flush=True)

    t0 = time.time()

    if which == "baseline":
        model, processor = build_baseline_model(device)
    else:
        model, processor = build_finetuned_model(device, args.best_dir)

    dtype = model.dtype
    print(f"  Model dtype: {dtype}")

    test_images = os.path.join(args.data_root, "processed/fgvc/images/test")
    test_labels = os.path.join(args.data_root, "test.csv")

    dataset = AircraftQADataset(
        csv_path=test_labels,
        images_path=test_images,
        processor=processor,
        train=False,        # no augmentation, deterministic
    )
    if args.limit:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  Test examples: {len(dataset)}  "
          f"(images × 3 questions; train images = unaugmented test set)")

    pred_csv = os.path.join(args.output_dir, f"{which}_predictions.csv")
    metrics = evaluate(model, loader, processor, device, dtype, which, pred_csv)

    json_path = os.path.join(args.output_dir, f"{which}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"  Saved metrics    → {json_path}")
    print(f"  Saved per-sample → {pred_csv}")
    print(f"  {which} elapsed: {elapsed_min:.1f} min")

    # Free GPU memory before next model.
    del model, processor, dataset, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Test InternVL3-2B (baseline + finetuned) on FGVC-Aircraft test set"
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Root containing processed/fgvc/images/test and test.csv")
    p.add_argument("--best_dir", type=str, required=True,
                   help="Path to best/ checkpoint directory (LoRA adapter)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write JSON metrics + per-sample prediction CSVs")
    p.add_argument("--which", type=str, default="both",
                   choices=["baseline", "finetuned", "both"])
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on examples for a smoke test (e.g. --limit 30)")
    return p.parse_args()


def preflight(args):
    """Check that everything we need exists; fail fast with a clear message."""
    test_images = os.path.join(args.data_root, "processed/fgvc/images/test")
    test_labels = os.path.join(args.data_root, "test.csv")

    problems = []
    if not os.path.isdir(test_images):
        problems.append(f"missing test images dir: {test_images}")
    elif len(os.listdir(test_images)) == 0:
        problems.append(f"empty test images dir: {test_images}")
    if not os.path.isfile(test_labels):
        problems.append(f"missing test labels: {test_labels}")

    if args.which in ("finetuned", "both"):
        if not os.path.isdir(args.best_dir):
            problems.append(f"missing best_dir: {args.best_dir}")
        elif not os.path.isfile(os.path.join(args.best_dir, "adapter_config.json")):
            problems.append(f"no adapter_config.json in {args.best_dir}")
        elif not os.path.isfile(os.path.join(args.best_dir, "adapter_model.safetensors")):
            problems.append(f"no adapter_model.safetensors in {args.best_dir}")

    if problems:
        print("[FATAL] preflight checks failed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"CUDA:         {torch.version.cuda}")
        print(f"bf16 OK:      {torch.cuda.is_bf16_supported()}")
    print(f"Data root:    {args.data_root}")
    print(f"Best dir:     {args.best_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Which:        {args.which}")
    print(f"Limit:        {args.limit}")
    print(f"Started:      {datetime.now().isoformat()}", flush=True)

    preflight(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.which == "both":
        models_to_run = ["baseline", "finetuned"]
    else:
        models_to_run = [args.which]

    summary = {}
    for which in models_to_run:
        try:
            metrics = run_model(which, device, args)
            if metrics is not None:
                summary[which] = metrics
        except Exception as e:
            print(f"\n[ERROR] {which} eval failed: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            # Free GPU memory and continue with the next model so one
            # failure doesn't lose the other model's results overnight.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # ── Comparison summary ──
    if len(summary) > 0:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        rows = [
            "test_loss", "test_ppl", "test_acc_overall",
            "test_acc_manufacturer", "test_acc_family", "test_acc_variant",
            "test_loss_manufacturer", "test_loss_family", "test_loss_variant",
            "test_ppl_manufacturer", "test_ppl_family", "test_ppl_variant",
        ]
        col_w = 28
        names = list(summary.keys())
        header = "metric".ljust(col_w) + "".join(n.ljust(16) for n in names)
        print(header)
        print("-" * len(header))
        for r in rows:
            line = r.ljust(col_w)
            for n in names:
                v = summary[n].get(r, "—")
                if isinstance(v, float):
                    line += f"{v:.4f}".ljust(16)
                else:
                    line += str(v).ljust(16)
            print(line)

    combined_path = os.path.join(args.output_dir, "summary.json")
    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCombined summary → {combined_path}")
    print(f"Done at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

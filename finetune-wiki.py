import math
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models import model_factory
from src.datasets.aircraft_text_dataset import AircraftTextDataset

from constants.constants import *


def _compute_metrics(outputs, labels):
    """Compute token-level accuracy and perplexity from model outputs and labels."""
    logits = outputs.logits
    preds = logits.argmax(dim=-1)

    # Shift: predict token t+1 from position t
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]

    # Only count non-padding tokens (where labels != -100)
    mask = shift_labels != -100
    correct = (shift_preds[mask] == shift_labels[mask]).sum().item()
    total = mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    perplexity = math.exp(outputs.loss.item())

    return accuracy, perplexity


def _plot_metrics(history, title_prefix=""):
    """Plot loss, accuracy, and perplexity across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title_prefix}Val Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["train_ppl"], label="Train")
    axes[2].plot(epochs, history["val_ppl"], label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Perplexity")
    axes[2].set_title(f"{title_prefix}Perplexity")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def finetune_wiki(model, train_loader, val_loader, num_epochs=3, grad_accum_steps=4, log_every=1):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
    )

    history = {"train_loss": [], "train_ppl": [],
               "val_loss": [], "val_acc": [], "val_ppl": []}

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        num_steps = len(train_loader)

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_steps:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()

            if (step + 1) % log_every == 0 or (step + 1) == num_steps:
                running_avg = train_loss / (step + 1)
                ppl = math.exp(running_avg)
                print(f"  [Train] Epoch {epoch+1} | Step {step+1}/{num_steps} | loss={running_avg:.4f} | ppl={ppl:.2f}")

        avg_train_loss = train_loss / num_steps
        train_ppl = math.exp(avg_train_loss)

        history["train_loss"].append(avg_train_loss)
        history["train_ppl"].append(train_ppl)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}

                outputs = model(**inputs)
                val_loss += outputs.loss.item()

                acc, _ = _compute_metrics(outputs, inputs["labels"])
                mask = inputs["labels"][:, 1:] != -100
                n = mask.sum().item()
                val_correct += acc * n
                val_total += n

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_ppl = math.exp(avg_val_loss)

        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)

        print(f"Epoch {epoch+1}/{num_epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}  val_ppl={val_ppl:.2f}")

    _plot_metrics(history, title_prefix="Wiki Finetune — ")

    return avg_val_loss, val_acc, val_ppl


def test(model, test_loader, log_every=10):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    num_steps = len(test_loader)

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**inputs)
            test_loss += outputs.loss.item()

            acc, _ = _compute_metrics(outputs, inputs["labels"])
            mask = inputs["labels"][:, 1:] != -100
            n = mask.sum().item()
            test_correct += acc * n
            test_total += n

            if (step + 1) % log_every == 0 or (step + 1) == num_steps:
                running_loss = test_loss / (step + 1)
                running_acc = test_correct / test_total if test_total > 0 else 0.0
                running_ppl = math.exp(running_loss)
                print(f"  [Test] Step {step+1}/{num_steps} | loss={running_loss:.4f} | acc={running_acc:.4f} | ppl={running_ppl:.2f}")

    avg_test_loss = test_loss / num_steps
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    test_ppl = math.exp(avg_test_loss)
    print(f"Test loss={avg_test_loss:.4f}  acc={test_acc:.4f}  ppl={test_ppl:.2f}")
    return avg_test_loss, test_acc, test_ppl


def sample_inference(model, processor, test_dataset):
    """Generate text from a prompt taken from the first test chunk."""
    model.eval()

    # Use first ~64 tokens of the first chunk as a prompt
    chunk = test_dataset.chunks[0]
    prompt_len = min(64, len(chunk) // 2)
    prompt_ids = chunk[:prompt_len]
    prompt_text = processor.tokenizer.decode(prompt_ids, skip_special_tokens=True)

    inputs = processor.tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    expected_text = processor.tokenizer.decode(chunk[prompt_len:prompt_len + 128], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt_text[:200]}...")
    print(f"Expected continuation: {expected_text[:200]}...")
    print(f"Generated: {generated_text[:200]}")
    print(f"{'='*60}\n")


def main():
    # --- Model ---
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
    )
    model.to(DEVICE_MPS)
    model.gradient_checkpointing_enable()

    # --- Datasets ---
    corpus_path = WIKI_CORPUS_EXPANDED  # Switch to WIKI_CORPUS_EXPANDED for the larger corpus

    train_dataset = AircraftTextDataset(
        jsonl_path=corpus_path, processor=processor, split="train",
    )
    val_dataset = AircraftTextDataset(
        jsonl_path=corpus_path, processor=processor, split="val",
    )
    test_dataset = AircraftTextDataset(
        jsonl_path=corpus_path, processor=processor, split="test",
    )

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}, Test chunks: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # --- Train ---
    finetune_wiki(model, train_loader, val_loader, num_epochs=3, grad_accum_steps=4, log_every=1)

    # --- Test ---
    #test(model, test_loader, log_every=10)

    # --- Sample inference ---
    #sample_inference(model, processor, test_dataset)


if __name__ == "__main__":
    main()

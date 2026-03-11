import torch
from torch.utils.data import DataLoader

from src.models import model_factory
from src.datasets.FGVC_aircraft_dataset import AircraftCaptionDataset

from constants.constants import *


def finetune_captions(model, train_loader, val_loader, num_epochs=3, grad_accum_steps=4, log_every=1):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
    )

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        num_steps = len(train_loader)

        for step, batch in enumerate(train_loader):
            # Move each tensor to the model's expected device
            inputs = {k: v.to(device=model.device, dtype=model.dtype) if k == "pixel_values"
                      else v.to(model.device)
                      for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_steps:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()

            if (step + 1) % log_every == 0 or (step + 1) == num_steps:
                running_avg = train_loss / (step + 1)
                print(f"  [Train] Epoch {epoch+1} | Step {step+1}/{num_steps} | running_loss={running_avg:.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device=model.device, dtype=model.dtype) if k == "pixel_values"
                          else v.to(model.device)
                          for k, v in batch.items()}

                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")


def main():
    # --- Model ---
    model, processor = model_factory.build_internvl3_2b(
        freeze_vision_encoder=True,
    )
    model.to(DEVICE_MPS)
    model.gradient_checkpointing_enable()

    # --- Datasets ---
    train_dataset = AircraftCaptionDataset(
        csv_path=FGVC_TRAIN_LABELS, images_path=FGVC_TRAIN_IMAGES, processor=processor,
    )
    val_dataset = AircraftCaptionDataset(
        csv_path=FGVC_VAL_LABELS, images_path=FGVC_VAL_IMAGES, processor=processor,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # --- Train ---
    finetune_captions(model, train_loader, val_loader, num_epochs=3, grad_accum_steps=4)


if __name__ == "__main__":
    main()

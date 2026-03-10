import torch
from torch.utils.data import DataLoader

from src.models import model_factory
from src.datasets.aircraft_text_dataset import AircraftTextDataset
from src.datasets.FGVC_aircraft_dataset import AircraftCaptionDataset

from constants.constants import *


def finetune_text(model, processor, optimizer, train_loader, val_loader, num_epochs=3):
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}

                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"[Text] Epoch {epoch+1}/{num_epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")


def finetune_captions(model, processor, optimizer, train_loader, val_loader, num_epochs=3):
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}

                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"[Captions] Epoch {epoch+1}/{num_epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")


def run_inference(model, processor, prompt, max_new_tokens=128):
    model.eval()

    inputs = processor.tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    return processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


def main():
    # --- Model ---
    model, processor = model_factory.build_internvl3_14b(
        freeze_vision_encoder=True,
        load_in_8bit=True,
        device_map=DEVICE_MPS
    )



    # --- Datasets ---
    #train_text_dataset = AircraftTextDataset()
    #val_text_dataset = AircraftTextDataset()

    train_caption_dataset = AircraftCaptionDataset(csv_path=FGVC_TRAIN_LABELS, images_path=FGVC_TRAIN_IMAGES, processor=processor)
    val_caption_dataset = AircraftCaptionDataset(csv_path=FGVC_VAL_LABELS, images_path=FGVC_VAL_IMAGES, processor=processor)

    #train_text_loader = DataLoader(train_text_dataset, batch_size=4, shuffle=True)
    #val_text_loader = DataLoader(val_text_dataset, batch_size=4)

    train_caption_loader = DataLoader(train_caption_dataset, batch_size=4, shuffle=True)
    val_caption_loader = DataLoader(val_caption_dataset, batch_size=4)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
    )

    # --- Stage 1: finetune on text ---
    #finetune_text(model, processor, optimizer, train_text_loader, val_text_loader, num_epochs=3)

    # --- Stage 2: finetune on captions ---
    finetune_captions(model, processor, optimizer, train_caption_loader, val_caption_loader, num_epochs=3)


if __name__ == "__main__":
    main()

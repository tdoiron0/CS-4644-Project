import torch

from src.models import model_factory
from src.datasets.aircraft_text_dataset import AircraftTextDataset
from src.datasets.FGVC_aircraft_dataset import AircraftCaptionDataset


def finetune_text(model, processor):
    model.train()



def finetune_captions(model, processor):
    model.train()


def run_inference(model, processor, prompt, max_new_tokens=128):
    model.eval()

    # Tokenize text only
    inputs = processor.tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Move to model device
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

    # initialize a MODEL_VLM with lora
    model, processor = model_factory.build_llavanext_lora()
    
    # print trainable parameters to verify lora setup
    #model.print_trainable_parameters()

    prompt = "Hello, what is your name?"

    run_inference(model, processor, prompt)


    

if __name__ == "__main__":
    main()
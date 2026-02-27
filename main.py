import torch

from src.models import model_factory
from src.datasets.aircraft_text_dataset import AircraftTextDataset
from src.datasets.FGVC_aircraft_dataset import AircraftCaptionDataset


def finetune_text():
    pass



def finetune_captions():
    pass


def main():

    # initialize a MODEL_VLM with lora, set mode to train
    model, processor = model_factory.build_vlm_lora()
    model.train()

    # print trainable parameters to verify lora setup
    model.print_trainable_parameters()




    

if __name__ == "__main__":
    main()
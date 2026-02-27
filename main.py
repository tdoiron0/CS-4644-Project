import torch

from src.models import model_factory


def main():

    # initialize a MODEL_VLM with lora, set mode to train
    model, processor = model_factory.build_vlm_lora()
    model.train()

    # print trainable parameters to verify lora setup
    model.print_trainable_parameters()




    

if __name__ == "__main__":
    main()
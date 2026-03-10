import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoModel
from peft import LoraConfig, get_peft_model

from constants.constants import *



def build_internvl3_14b(
    freeze_vision_encoder: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

    model = AutoModel.from_pretrained(
        MODEL_INTERNVL3_14B,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_INTERNVL3_14B,
        trust_remote_code=True,
    )

    if freeze_vision_encoder:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def build_internvl3_8b(
    freeze_vision_encoder: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

    model = AutoModel.from_pretrained(
        MODEL_INTERNVL3_8B,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_INTERNVL3_8B,
        trust_remote_code=True,
    )

    if freeze_vision_encoder:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor

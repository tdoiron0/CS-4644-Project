import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

from constants.constants import *


def _build_internvl3(
    model_name,
    freeze_vision_encoder=True,
    train_projector=True,
    load_in_8bit=False,
    device_map=None,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
):
    if torch.backends.mps.is_available():
        dtype = torch.float32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    kwargs = {
        "torch_dtype": dtype,
    }

    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = device_map or "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

    processor = AutoProcessor.from_pretrained(model_name)

    if freeze_vision_encoder:
        for param in model.model.vision_tower.parameters():
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

    if train_projector:
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name:
                param.requires_grad = True

    model.print_trainable_parameters()

    return model, processor


def build_internvl3_14b(**kwargs):
    return _build_internvl3(MODEL_INTERNVL3_14B, **kwargs)


def build_internvl3_8b(**kwargs):
    return _build_internvl3(MODEL_INTERNVL3_8B, **kwargs)


def build_internvl3_2b(**kwargs):
    return _build_internvl3(MODEL_INTERNVL3_2B, **kwargs)
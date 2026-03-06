import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from constants.constants import *


def build_llavanext_lora():
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_LLAVA_NEXT,
        torch_dtype=torch.float16,   # or bfloat16 if supported in your env
    )

    model.to(DEVICE_MPS)

    processor = AutoProcessor.from_pretrained(MODEL_LLAVA_NEXT)  # must match

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    model = get_peft_model(model, lora_config)

    return model, processor


def build_llavanext_lora_cuda():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_LLAVA_NEXT,
        quantization_config=bnb_config,
        device_map=DEVICE_CUDA
        )

    processor = AutoProcessor.from_pretrained(MODEL_LLAVA_NEXT)  # must match


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    model = get_peft_model(model, lora_config)

    return model, processor


def build_internvl3(
    freeze_vision_encoder: bool = False,
    freeze_llm: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

    model = AutoModel.from_pretrained(
        MODEL_INTERNVL3,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_INTERNVL3,
        trust_remote_code=True,
    )

    if freeze_vision_encoder:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if freeze_llm:
        for param in model.language_model.parameters():
            param.requires_grad = False

    return model, tokenizer

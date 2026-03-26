"""
Model setup module for QLoRA fine-tuning.

Handles BitsAndBytes 4-bit quantization config, model loading,
LoRA adapter injection, and parameter counting.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def get_quantization_config() -> BitsAndBytesConfig:
    """Create 4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA adapter configuration."""
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config: BitsAndBytesConfig | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model with 4-bit quantization and its tokenizer.
    """
    if quantization_config is None:
        quantization_config = get_quantization_config()

    print(f"Loading model: {model_name}")
    print(f"Quantization: 4-bit NF4 with double quantization")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Stable for QLoRA training
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token (Llama doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"

    print(f"Model loaded on: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig | None = None,
) -> AutoModelForCausalLM:
    """
    Prepare model for QLoRA training and inject LoRA adapters.

    Returns the PEFT model with adapter layers.
    """
    if lora_config is None:
        lora_config = get_lora_config()

    # Prepare for k-bit training (handles gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # Inject LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print parameter stats
    print_trainable_parameters(model)

    return model


def print_trainable_parameters(model) -> dict:
    """
    Print and return the number of trainable vs total parameters.
    """
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    reduction_pct = 100 * (1 - trainable / total)

    stats = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "reduction_pct": reduction_pct,
    }

    print(f"\n{'='*50}")
    print(f"PARAMETER SUMMARY")
    print(f"{'='*50}")
    print(f"Total parameters:     {total:>14,}")
    print(f"Trainable parameters: {trainable:>14,}")
    print(f"Trainable %:          {stats['trainable_pct']:>13.2f}%")
    print(f"Parameter reduction:  {reduction_pct:>13.1f}%")
    print(f"{'='*50}\n")

    return stats

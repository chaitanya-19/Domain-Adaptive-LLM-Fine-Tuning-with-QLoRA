"""
End-to-end pipeline: Data Prep → QLoRA Fine-Tuning → Evaluation

Usage (Google Colab or local):
    python scripts/run_pipeline.py --config configs/config.yaml

Set environment variables before running:
    export HF_TOKEN=<your_huggingface_token>
    export OPENAI_API_KEY=<your_openai_key>  # For RAGAS evaluation
"""

import argparse
import os
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import prepare_dataset, prepare_evaluation_samples
from src.model_setup import (
    get_quantization_config,
    get_lora_config,
    load_model_and_tokenizer,
    apply_lora,
)
from src.train import get_training_args, create_trainer, train_model
from src.inference import (
    load_finetuned_model,
    batch_generate,
    compare_models,
    display_comparison,
)
from src.evaluate import (
    setup_ragas_llm,
    full_evaluation,
    print_results,
    save_results,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict):
    """Execute the full fine-tuning and evaluation pipeline."""

    # ========================================
    # STAGE 1: Load Model & Tokenizer
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 1: Loading Model & Tokenizer")
    print("=" * 60)

    bnb_config = get_quantization_config()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        quantization_config=bnb_config,
    )

    # ========================================
    # STAGE 2: Prepare Dataset
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 2: Preparing Dataset")
    print("=" * 60)

    ds_cfg = config["dataset"]
    train_dataset, eval_dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_name=ds_cfg["name"],
        system_prompt=config["system_prompt"],
        max_samples=ds_cfg.get("max_samples"),
        train_split_ratio=ds_cfg["train_split_ratio"],
        seed=ds_cfg["seed"],
    )

    # ========================================
    # STAGE 3: Apply QLoRA
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 3: Applying QLoRA Adapters")
    print("=" * 60)

    lora_cfg = config["lora"]
    lora_config = get_lora_config(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = apply_lora(model, lora_config)

    # ========================================
    # STAGE 4: Training
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 4: Fine-Tuning with QLoRA")
    print("=" * 60)

    train_cfg = config["training"]
    training_args = get_training_args(**train_cfg)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        max_seq_length=config["model"]["max_seq_length"],
    )

    train_metrics = train_model(trainer, output_dir=train_cfg["output_dir"])

    # Free memory before evaluation
    del model
    del trainer
    torch.cuda.empty_cache()

    # ========================================
    # STAGE 5: Load Models for Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 5: Loading Models for Evaluation")
    print("=" * 60)

    # Load base model (fresh, no adapter)
    base_model, base_tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
    )
    base_model.eval()

    # Load fine-tuned model
    ft_model, ft_tokenizer = load_finetuned_model(
        base_model_name=config["model"]["name"],
        adapter_path=train_cfg["output_dir"],
    )

    # ========================================
    # STAGE 6: Generate Answers
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 6: Generating Answers for Evaluation")
    print("=" * 60)

    eval_cfg = config["evaluation"]
    eval_samples = prepare_evaluation_samples(
        dataset_name=ds_cfg["name"],
        num_samples=eval_cfg["num_samples"],
        seed=ds_cfg["seed"],
    )

    questions = [s["question"] for s in eval_samples]
    ground_truths = [s["ground_truth"] for s in eval_samples]
    contexts = [s["context"] for s in eval_samples]

    print("\nGenerating base model answers...")
    base_answers = batch_generate(
        base_model, base_tokenizer, questions,
        system_prompt=config["system_prompt"],
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
    )

    print("\nGenerating fine-tuned model answers...")
    ft_answers = batch_generate(
        ft_model, ft_tokenizer, questions,
        system_prompt=config["system_prompt"],
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
    )

    # Qualitative comparison
    comparison = [
        {"question": q, "base_answer": ba, "finetuned_answer": fa}
        for q, ba, fa in zip(questions, base_answers, ft_answers)
    ]
    display_comparison(comparison, num_display=3)

    # ========================================
    # STAGE 7: RAGAS + Traditional Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 7: Running RAGAS Evaluation")
    print("=" * 60)

    ragas_llm, ragas_embeddings = setup_ragas_llm(
        model_name=eval_cfg.get("ragas_llm", "gpt-4o-mini"),
        embedding_model=eval_cfg.get("ragas_embeddings", "text-embedding-3-small"),
    )

    results = full_evaluation(
        questions=questions,
        base_answers=base_answers,
        ft_answers=ft_answers,
        ground_truths=ground_truths,
        contexts=contexts,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    # Add training metrics
    results["training_metrics"] = train_metrics

    # Print and save
    print_results(results)
    save_results(results, output_dir=train_cfg["output_dir"])

    print("\n✅ Pipeline complete!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive LLM Fine-Tuning Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_pipeline(config)
    return results


if __name__ == "__main__":
    main()

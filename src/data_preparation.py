"""
Data preparation module for medical Q&A fine-tuning.

Loads the medalpaca/medical_meadow_medical_flashcards dataset,
formats it into Llama 3.2 chat template, and splits into train/eval.
"""

from datasets import load_dataset, Dataset
from typing import Optional


def load_medical_dataset(
    dataset_name: str = "medalpaca/medical_meadow_medical_flashcards",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load the medical flashcards dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        print(f"Sampled {max_samples} examples from {dataset_name}")

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    return dataset


def format_chat_template(example: dict, tokenizer, system_prompt: str) -> dict:
    """
    Format a single example into Llama 3.2 chat template.

    The medical flashcards dataset has 'input' (question) and 'output' (answer).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Apply the model's chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": formatted}


def prepare_dataset(
    tokenizer,
    dataset_name: str = "medalpaca/medical_meadow_medical_flashcards",
    system_prompt: str = "You are a knowledgeable medical assistant.",
    max_samples: Optional[int] = None,
    train_split_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    End-to-end dataset preparation pipeline.

    Returns:
        (train_dataset, eval_dataset)
    """
    # Load raw dataset
    dataset = load_medical_dataset(dataset_name, max_samples, seed)

    # Format with chat template
    print("Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda ex: format_chat_template(ex, tokenizer, system_prompt),
        remove_columns=dataset.column_names,
        desc="Applying chat template",
    )

    # Train/eval split
    split = dataset.train_test_split(test_size=1 - train_split_ratio, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")

    # Preview one example
    print("\n--- Sample formatted example ---")
    print(train_dataset[0]["text"][:600])
    print("...")

    return train_dataset, eval_dataset


def prepare_evaluation_samples(
    dataset_name: str = "medalpaca/medical_meadow_medical_flashcards",
    num_samples: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Prepare held-out samples for RAGAS evaluation.

    Returns list of dicts with 'question', 'ground_truth', and 'context'.
    """
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed + 100)  # Different seed from training

    # Take samples from the tail end to avoid overlap with training data
    eval_samples = dataset.select(range(len(dataset) - num_samples, len(dataset)))

    samples = []
    for ex in eval_samples:
        samples.append({
            "question": ex["input"],
            "ground_truth": ex["output"],
            # Context = ground truth for faithfulness evaluation
            # In a RAG setup, this would come from a retriever
            "context": ex["output"],
        })

    print(f"Prepared {len(samples)} evaluation samples")
    return samples

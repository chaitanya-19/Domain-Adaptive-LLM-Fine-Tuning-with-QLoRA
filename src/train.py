"""
Training module for QLoRA fine-tuning using TRL's SFTTrainer.
"""

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset


def get_training_args(
    output_dir: str = "./results/qlora-medical-llama3.2-3b",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 25,
    eval_steps: int = 100,
    save_steps: int = 200,
    save_total_limit: int = 2,
    max_grad_norm: float = 0.3,
    optim: str = "paged_adamw_32bit",
    **kwargs,
) -> TrainingArguments:
    """Create training arguments for SFTTrainer."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        max_grad_norm=max_grad_norm,
        optim=optim,
        group_by_length=True,
        report_to="none",
        remove_unused_columns=False,
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments | None = None,
    max_seq_length: int = 1024,
) -> SFTTrainer:
    """
    Create the SFTTrainer instance for supervised fine-tuning.
    """
    if training_args is None:
        training_args = get_training_args()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    print(f"Trainer created successfully")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    print(f"  Epochs:             {training_args.num_train_epochs}")
    print(f"  Batch size:         {training_args.per_device_train_batch_size}")
    print(f"  Grad accumulation:  {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch:    {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate:      {training_args.learning_rate}")
    print(f"  Max seq length:     {max_seq_length}")

    return trainer


def train_model(trainer: SFTTrainer, output_dir: str = None) -> dict:
    """
    Execute training and save the adapter weights.

    Returns training metrics.
    """
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50 + "\n")

    # Train
    train_result = trainer.train()

    # Save adapter weights
    save_dir = output_dir or trainer.args.output_dir
    print(f"\nSaving adapter weights to: {save_dir}")
    trainer.model.save_pretrained(save_dir)
    trainer.tokenizer.save_pretrained(save_dir)

    # Collect metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }

    # Evaluate on eval set
    eval_metrics = trainer.evaluate()
    metrics["eval_loss"] = eval_metrics.get("eval_loss", 0)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Train loss:       {metrics['train_loss']:.4f}")
    print(f"Eval loss:        {metrics['eval_loss']:.4f}")
    print(f"Training time:    {metrics['train_runtime']:.1f}s")
    print(f"Samples/sec:      {metrics['train_samples_per_second']:.1f}")
    print(f"Adapter saved to: {save_dir}")

    return metrics

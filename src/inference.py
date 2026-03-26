"""
Inference module for generating answers from base and fine-tuned models.
Used for qualitative comparison and feeding into RAGAS evaluation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model with merged LoRA adapter weights.
    """
    print(f"Loading base model: {base_model_name}")
    print(f"Loading adapter from: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Fine-tuned model loaded successfully")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    system_prompt: str = "You are a knowledgeable medical assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer for a single question using the chat template.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (exclude prompt tokens)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return answer


def batch_generate(
    model,
    tokenizer,
    questions: list[str],
    system_prompt: str = "You are a knowledgeable medical assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> list[str]:
    """
    Generate answers for a batch of questions sequentially.
    """
    answers = []
    for q in tqdm(questions, desc="Generating answers"):
        answer = generate_answer(
            model, tokenizer, q,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answers.append(answer)
    return answers


def compare_models(
    base_model,
    base_tokenizer,
    ft_model,
    ft_tokenizer,
    questions: list[str],
    system_prompt: str = "You are a knowledgeable medical assistant.",
    max_new_tokens: int = 256,
) -> list[dict]:
    """
    Generate answers from both models for side-by-side comparison.
    """
    results = []

    for q in tqdm(questions, desc="Comparing models"):
        base_answer = generate_answer(
            base_model, base_tokenizer, q,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        ft_answer = generate_answer(
            ft_model, ft_tokenizer, q,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        results.append({
            "question": q,
            "base_answer": base_answer,
            "finetuned_answer": ft_answer,
        })

    return results


def display_comparison(results: list[dict], num_display: int = 5):
    """Pretty-print a side-by-side comparison of model outputs."""
    for i, r in enumerate(results[:num_display]):
        print(f"\n{'='*70}")
        print(f"QUESTION {i+1}: {r['question'][:200]}")
        print(f"{'='*70}")
        print(f"\n[BASE MODEL]:")
        print(r["base_answer"][:500])
        print(f"\n[FINE-TUNED]:")
        print(r["finetuned_answer"][:500])
        print()

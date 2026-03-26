"""
Evaluation module using RAGAS metrics.

Benchmarks both base and fine-tuned models on:
- Faithfulness: Is the answer faithful to the provided context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: How precise is the context for answering?

Also includes traditional NLP metrics (ROUGE, BLEU) for completeness.
"""

import json
import os
from datetime import datetime

import numpy as np
from datasets import Dataset

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain wrappers for RAGAS judge
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Traditional metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


def setup_ragas_llm(
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
):
    """
    Initialize the LLM judge and embeddings for RAGAS evaluation.
    Requires OPENAI_API_KEY environment variable.
    """
    llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    return llm, embeddings


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[str],
    ground_truths: list[str],
    llm=None,
    embeddings=None,
) -> dict:
    """
    Run RAGAS evaluation on a set of generated answers.

    Args:
        questions:     List of questions
        answers:       List of model-generated answers
        contexts:      List of reference contexts (ground truth used as context)
        ground_truths: List of ground truth answers
        llm:           RAGAS LLM judge (optional, auto-created if None)
        embeddings:    RAGAS embeddings (optional, auto-created if None)

    Returns:
        Dictionary with metric scores
    """
    if llm is None or embeddings is None:
        llm, embeddings = setup_ragas_llm()

    # Build RAGAS-compatible dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": [[ctx] for ctx in contexts],  # RAGAS expects list of lists
        "ground_truth": ground_truths,
    })

    print(f"Running RAGAS evaluation on {len(questions)} samples...")

    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
    }

    return scores


def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_bleu_scores(predictions: list[str], references: list[str]) -> float:
    """Compute average BLEU score."""
    nltk.download("punkt_tab", quiet=True)
    smoother = SmoothingFunction().method1

    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())

        if len(pred_tokens) == 0:
            scores.append(0.0)
            continue

        score = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=smoother,
        )
        scores.append(score)

    return float(np.mean(scores))


def full_evaluation(
    questions: list[str],
    base_answers: list[str],
    ft_answers: list[str],
    ground_truths: list[str],
    contexts: list[str] | None = None,
    ragas_llm=None,
    ragas_embeddings=None,
) -> dict:
    """
    Run complete evaluation comparing base and fine-tuned models.

    Returns a comprehensive results dictionary.
    """
    if contexts is None:
        contexts = ground_truths  # Use ground truth as context

    results = {"timestamp": datetime.now().isoformat(), "num_samples": len(questions)}

    # ---- RAGAS Evaluation ----
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION - BASE MODEL")
    print("=" * 50)
    base_ragas = run_ragas_evaluation(
        questions, base_answers, contexts, ground_truths,
        llm=ragas_llm, embeddings=ragas_embeddings,
    )
    results["base_model"] = {"ragas": base_ragas}

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION - FINE-TUNED MODEL")
    print("=" * 50)
    ft_ragas = run_ragas_evaluation(
        questions, ft_answers, contexts, ground_truths,
        llm=ragas_llm, embeddings=ragas_embeddings,
    )
    results["finetuned_model"] = {"ragas": ft_ragas}

    # ---- Traditional Metrics ----
    print("\nComputing ROUGE and BLEU scores...")

    base_rouge = compute_rouge_scores(base_answers, ground_truths)
    ft_rouge = compute_rouge_scores(ft_answers, ground_truths)

    base_bleu = compute_bleu_scores(base_answers, ground_truths)
    ft_bleu = compute_bleu_scores(ft_answers, ground_truths)

    results["base_model"]["rouge"] = base_rouge
    results["base_model"]["bleu"] = base_bleu
    results["finetuned_model"]["rouge"] = ft_rouge
    results["finetuned_model"]["bleu"] = ft_bleu

    # ---- Compute Improvements ----
    improvements = {}
    for metric in base_ragas:
        base_val = base_ragas[metric]
        ft_val = ft_ragas[metric]
        if base_val > 0:
            pct_change = ((ft_val - base_val) / base_val) * 100
        else:
            pct_change = 0.0
        improvements[f"ragas_{metric}"] = round(pct_change, 2)

    for metric in base_rouge:
        base_val = base_rouge[metric]
        ft_val = ft_rouge[metric]
        if base_val > 0:
            pct_change = ((ft_val - base_val) / base_val) * 100
        else:
            pct_change = 0.0
        improvements[metric] = round(pct_change, 2)

    if base_bleu > 0:
        improvements["bleu"] = round(((ft_bleu - base_bleu) / base_bleu) * 100, 2)

    results["improvements_pct"] = improvements

    return results


def print_results(results: dict):
    """Pretty-print the evaluation results."""
    print(f"\n{'='*70}")
    print(f"{'EVALUATION RESULTS':^70}")
    print(f"{'='*70}")
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Timestamp: {results['timestamp']}")

    # RAGAS
    print(f"\n{'--- RAGAS Metrics ---':^70}")
    print(f"{'Metric':<25} {'Base':>12} {'Fine-tuned':>12} {'Change':>12}")
    print("-" * 65)
    for metric in results["base_model"]["ragas"]:
        bv = results["base_model"]["ragas"][metric]
        fv = results["finetuned_model"]["ragas"][metric]
        ch = results["improvements_pct"].get(f"ragas_{metric}", 0)
        arrow = "↑" if ch > 0 else "↓" if ch < 0 else "→"
        print(f"{metric:<25} {bv:>12.4f} {fv:>12.4f} {arrow} {ch:>+.1f}%")

    # ROUGE / BLEU
    print(f"\n{'--- Traditional Metrics ---':^70}")
    print(f"{'Metric':<25} {'Base':>12} {'Fine-tuned':>12} {'Change':>12}")
    print("-" * 65)
    for metric in results["base_model"]["rouge"]:
        bv = results["base_model"]["rouge"][metric]
        fv = results["finetuned_model"]["rouge"][metric]
        ch = results["improvements_pct"].get(metric, 0)
        arrow = "↑" if ch > 0 else "↓" if ch < 0 else "→"
        print(f"{metric:<25} {bv:>12.4f} {fv:>12.4f} {arrow} {ch:>+.1f}%")

    bv = results["base_model"]["bleu"]
    fv = results["finetuned_model"]["bleu"]
    ch = results["improvements_pct"].get("bleu", 0)
    arrow = "↑" if ch > 0 else "↓" if ch < 0 else "→"
    print(f"{'bleu':<25} {bv:>12.4f} {fv:>12.4f} {arrow} {ch:>+.1f}%")

    print(f"\n{'='*70}\n")


def save_results(results: dict, output_dir: str = "./results"):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {path}")
    return path

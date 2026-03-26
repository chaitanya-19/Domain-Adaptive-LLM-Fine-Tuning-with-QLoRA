# 🏥 Domain-Adaptive LLM Fine-Tuning with QLoRA

Fine-tuned **Llama 3.2 3B** for domain-specific **medical Q&A** using **QLoRA** (4-bit NF4 quantization + LoRA adapters), reducing trainable parameters by **~90%** while preserving model performance on domain tasks.

Built an end-to-end **LLM evaluation pipeline using RAGAS** to benchmark the fine-tuned model across faithfulness, answer relevancy, and context precision — along with traditional NLP metrics (ROUGE, BLEU).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/domain-adaptive-llm-finetuning/blob/main/notebooks/Domain_Adaptive_LLM_FineTuning.ipynb)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Medical Flashcards    Llama 3.2 3B       QLoRA Config   │
│  (33K Q&A pairs)       (4-bit NF4)       (r=16, α=32)   │
│        │                    │                  │          │
│        └────────┬───────────┘──────────────────┘          │
│                 │                                         │
│          ┌──────▼──────┐                                  │
│          │ SFTTrainer  │  • Paged AdamW 32-bit            │
│          │ (TRL)       │  • Cosine LR schedule            │
│          │             │  • Gradient accumulation          │
│          └──────┬──────┘                                  │
│                 │                                         │
│          ┌──────▼──────┐                                  │
│          │   Adapter   │  ~3-5% trainable params          │
│          │   Weights   │  LoRA on all linear layers       │
│          └─────────────┘                                  │
├──────────────────────────────────────────────────────────┤
│                   Evaluation Pipeline                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Base Model ──► Generate Answers ──┐                     │
│                                     ├──► RAGAS Eval      │
│  Fine-tuned ──► Generate Answers ──┘    • Faithfulness   │
│                                         • Relevancy      │
│                                         • Precision      │
│                                                          │
│                                    ──► Traditional       │
│                                         • ROUGE-1/2/L    │
│                                         • BLEU           │
└──────────────────────────────────────────────────────────┘
```

## Key Results

| Metric | Base Model | Fine-Tuned | Change |
|---|---|---|---|
| **Faithfulness** | — | — | ↑ |
| **Answer Relevancy** | — | — | ↑ |
| **Context Precision** | — | — | ↑ |
| **ROUGE-L** | — | — | ↑ |
| **BLEU** | — | — | ↑ |

> *Fill in with your actual results after running the pipeline*

### Parameter Efficiency

| | Count |
|---|---|
| Total Parameters | ~3.2B |
| Trainable (LoRA) | ~160M |
| **Reduction** | **~95%** |

## Tech Stack

- **Model:** Meta Llama 3.2 3B Instruct
- **Quantization:** BitsAndBytes 4-bit NF4 with double quantization
- **Fine-tuning:** QLoRA via HuggingFace PEFT + TRL SFTTrainer
- **Dataset:** [Medical Meadow Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) (33K pairs)
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision) + ROUGE + BLEU
- **Compute:** Google Colab T4 GPU (16GB VRAM)

## Project Structure

```
├── configs/
│   └── config.yaml                 # All hyperparameters
├── notebooks/
│   └── Domain_Adaptive_LLM_FineTuning.ipynb  # Full Colab notebook
├── scripts/
│   └── run_pipeline.py             # End-to-end CLI pipeline
├── src/
│   ├── data_preparation.py         # Dataset loading & formatting
│   ├── model_setup.py              # BitsAndBytes + LoRA config
│   ├── train.py                    # SFTTrainer training loop
│   ├── inference.py                # Answer generation & comparison
│   └── evaluate.py                 # RAGAS + ROUGE/BLEU evaluation
├── results/                        # Training artifacts & metrics
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Add your tokens in Colab Secrets (sidebar → 🔑):
   - `HF_TOKEN` — [Get from HuggingFace](https://huggingface.co/settings/tokens)
   - `OPENAI_API_KEY` — [Get from OpenAI](https://platform.openai.com/api-keys)
4. **Accept** the [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on HuggingFace
5. Run all cells

### Option 2: CLI Pipeline

```bash
git clone https://github.com/YOUR_USERNAME/domain-adaptive-llm-finetuning.git
cd domain-adaptive-llm-finetuning

pip install -r requirements.txt

export HF_TOKEN=<your_token>
export OPENAI_API_KEY=<your_key>

python scripts/run_pipeline.py --config configs/config.yaml
```

## Technical Deep Dive

### Why QLoRA?

Full fine-tuning of a 3B parameter model requires ~24GB+ VRAM and updates all parameters. QLoRA makes this feasible on consumer hardware by:

1. **4-bit NF4 Quantization** — Compresses base model weights from FP16 to 4-bit using a normal-float data type optimized for neural network weight distributions
2. **Double Quantization** — Quantizes the quantization constants themselves, saving ~0.4 bits per parameter
3. **LoRA Adapters** — Injects low-rank trainable matrices (rank=16) into attention and MLP layers, training only ~3-5% of total parameters
4. **Paged Optimizers** — Uses CPU memory offloading for optimizer states to prevent OOM

### Training Configuration

| Setting | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Balance between capacity and efficiency |
| LoRA alpha | 32 | 2x rank, standard scaling factor |
| Target modules | All linear | Attention (Q,K,V,O) + MLP (gate, up, down) |
| Batch size | 4 × 4 grad accum | Effective batch of 16 |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Scheduler | Cosine | Smooth decay, better convergence |
| Epochs | 3 | Enough for domain adaptation without overfitting |

### RAGAS Evaluation

[RAGAS](https://docs.ragas.io/) provides LLM-based evaluation metrics:

- **Faithfulness** — Does the generated answer contain only information present in the provided context? Measures hallucination.
- **Answer Relevancy** — Is the answer actually addressing the question asked? Penalizes vague or off-topic responses.
- **Context Precision** — How precise is the reference context for answering the question?

The evaluation uses GPT-4o-mini as the judge model and text-embedding-3-small for embeddings.

## Configuration

All hyperparameters are in `configs/config.yaml`. Key knobs to tune:

```yaml
# Use fewer samples for faster iteration
dataset:
  max_samples: 5000    # Default: 10000, null for all 33K

# Adjust LoRA capacity
lora:
  r: 8                 # Lower = fewer params, faster training
  lora_alpha: 16       # Keep 2x rank

# Adjust training
training:
  num_train_epochs: 1  # Quick test run
  learning_rate: 1e-4  # Lower for more stability
```

## License

This project is for educational and research purposes. The Llama 3.2 model is subject to [Meta's Llama License](https://llama.meta.com/llama3/license/). The medical dataset is from [MedAlpaca](https://github.com/kbressem/medAlpaca).

⚠️ **Disclaimer:** This model is NOT intended for clinical use. Always consult qualified healthcare professionals for medical decisions.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contrastive fine-tuning (CFT) of small language models (up to 2B params) for text embedding, using InfoNCE loss with LoRA. The project trains models to produce better sentence embeddings for Semantic Textual Similarity (STS) tasks, evaluated via MTEB benchmarks.

## Environment Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .   # or: uv pip install -r pyproject.toml
```

Dependencies are declared in `pyproject.toml`. Python 3.11, PyTorch 2.2, Transformers 4.40, PEFT 0.10. Requires CUDA GPUs for training.

## Pretrained Model

The `pretrained/` directory is gitignored. Download the full model (required for data prep, training, and evaluation):

```bash
huggingface-cli download openbmb/MiniCPM-2B-dpo-bf16 \
  --local-dir pretrained/MiniCPM-2B-dpo-bf16
```

Verify `add_eos_token` is set (required — embeddings use the EOS token):
```bash
grep add_eos_token pretrained/MiniCPM-2B-dpo-bf16/tokenizer_config.json
# should show: "add_eos_token": true,
```

> **Note:** For data preprocessing only, the tokenizer files suffice: `huggingface-cli download openbmb/MiniCPM-2B-dpo-bf16 tokenizer.model tokenizer_config.json special_tokens_map.json --local-dir pretrained/MiniCPM-2B-dpo-bf16`

## Commands

### Data preparation
```bash
cd data
./download_nli.sh          # downloads nli_for_simcse.csv from HuggingFace
python nli_preprocess.py   # tokenizes and saves to data/processed/
```

Pilot mode (smaller dataset for quick iteration):
```bash
python nli_preprocess.py --num_rows 1000   # saves to data/processed_pilot/
```

Note: `sentencepiece` is required for the MiniCPM (LLaMA-based) tokenizer and is included in `pyproject.toml`.

### Training (multi-GPU with DDP via Accelerate)
```bash
cd train
# Configure GPU count in configs/ddp_config.yaml (num_processes field)
./train.sh
```

The train script uses `accelerate launch` with the DDP config. Output adapters are saved to `train/output/<timestamp>/`.

### Evaluation (MTEB benchmarks)
```bash
cd eval/mteb
python minicpm_sts_eval.py        # STS benchmarks
python minicpm_retrieval_eval.py  # Retrieval benchmarks
```

Results saved to `eval/mteb/results/minicpm/`.

## Architecture

### Training pipeline (`train/`)

- **`train.py`** — Entry point. Parses `ModelArguments`, `DataArguments`, and custom `TrainingArguments` (adds `temperature` param) via HfArgumentParser. Loads a causal LM, wraps it with LoRA via PEFT, loads preprocessed dataset, and runs `ContrastiveTrainer`.
- **`contrastive_trainer.py`** — Subclass of HuggingFace `Trainer`. Overrides `compute_loss` to encode three inputs (anchor/`sent0`, positive/`sent1`, hard negative/`hard_neg`) by extracting the last hidden state at the final token position, then passes embeddings to InfoNCE loss.
- **`loss.py`** — `InfoNCE` module. Normalizes embeddings, uses `AllGather` across GPUs for global batch negatives, computes cosine similarity logits, and applies cross-entropy with temperature scaling.
- **`utils.py`** — Custom `AllGather` autograd function for gradient-enabled all-gather across distributed processes.

### Data (`data/`)

- **`nli_preprocess.py`** — Tokenizes NLI triplets (sent0, sent1, hard_neg) with the MiniCPM tokenizer, padding to max_length=150. Saves as HuggingFace dataset to `data/processed/` (or `data/processed_pilot/` with `--num_rows`). Expects pretrained model at `pretrained/MiniCPM-2B-dpo-bf16/`.

### Evaluation (`eval/mteb/`)

- **`model/minicpm.py`** — `MiniCPM` wrapper class with `encode()` method for MTEB compatibility. Extracts embeddings from the last hidden state of the final token. Optionally loads a LoRA adapter.
- Eval scripts expect model at `pretrained/MiniCPM-2B-dpo-bf16` and adapter at `pretrained/adapter/<timestamp>/`.

### Key design decisions

- Embeddings are extracted from the **last token** (EOS) of the **last hidden layer** — the tokenizer must have `add_eos_token: true` set in `tokenizer_config.json`.
- LoRA targets `q_proj` and `v_proj` by default (rank 8, alpha 32, dropout 0.1).
- In-batch negatives are combined with explicit hard negatives in the InfoNCE loss. `AllGather` enables using negatives across all GPUs for a larger effective batch.
- Pretrained models are expected at `pretrained/` (gitignored). Trained adapters go to `train/output/` (also gitignored).

## Remote Training Practices

When running training or evaluation on a remote GPU instance (Lambda Cloud, etc.), always follow these conventions:

1. **Always use `tmux`** — Every SSH session to a remote machine must start inside a `tmux` session so that long-running jobs survive disconnects.

   ```bash
   tmux new -s training
   # or reattach: tmux attach -t training
   ```

2. **Always use `tee` for log capture** — Pipe stdout/stderr to a timestamped log file while still displaying output in the terminal.

   ```bash
   ./train.sh 2>&1 | tee train_$(date +%Y%m%d%H%M%S).log
   ```

3. **Always send a push notification on completion** — Append an `ntfy.sh` curl after every long-running command so the user gets notified on their iPhone.

   ```bash
   ./train.sh 2>&1 | tee train.log; \
     curl -d "Training finished (exit code: $?)" ntfy.sh/ntfy.sh/LM-STS-CFT
   ```

When composing remote commands, combine all three:

```bash
tmux new -s training
# then inside tmux:
./train.sh 2>&1 | tee train_$(date +%Y%m%d%H%M%S).log; \
  curl -d "Training finished (exit code: $?)" ntfy.sh/ntfy.sh/LM-STS-CFT
```

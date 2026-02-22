# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contrastive fine-tuning (CFT) of small language models (up to 2B params) for text embedding, using InfoNCE loss with LoRA. The project trains models to produce better sentence embeddings for Semantic Textual Similarity (STS) tasks, evaluated via MTEB benchmarks and custom Sanskrit STS evaluation.

## Environment Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Dependencies are declared in `pyproject.toml`. Python 3.11, PyTorch 2.2, Transformers 4.40, PEFT 0.10. Requires CUDA GPUs for training.

## Pretrained Models

The `pretrained/` directory is gitignored. Two backbones are supported:

### Sarvam-1 (primary — Sanskrit-capable)

```bash
cd data && ./download_sarvam.sh
```

This downloads `sarvamai/sarvam-1` to `pretrained/sarvam-1/` and sets `add_eos_token: true` in the tokenizer config. Sarvam-1 tokenizes Sanskrit at ~3.9x fertility with meaningful Devanagari subwords.

### MiniCPM-2B (legacy — English)

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

Custom dataset with Sarvam-1 tokenizer:
```bash
python nli_preprocess.py --input_csv saiva_triplets.csv \
  --tokenizer_path ../pretrained/sarvam-1/ --output_dir ./processed_shaiva/
```

All preprocessing args: `--tokenizer_path`, `--max_length`, `--input_csv`, `--output_dir`, `--num_rows`. Run `--help` for details.

#### Itihasa triplets (Stage 1 — general Sanskrit)

```bash
cd data
python itihasa_triplets.py                             # ~167K cross-lingual + monolingual Sa triplets
python itihasa_triplets.py --num_rows 1000             # pilot mode
python itihasa_triplets.py --triplet_types mono_sa     # Sanskrit-only triplets
```

Downloads `rahular/itihasa` (93K Sanskrit-English parallel pairs from Ramayana + Mahabharata). Generates cross-lingual triplets `(sn, en, distant_en)` and monolingual Sanskrit triplets `(sn[i], sn[i+1], distant_sn)`. Hard negatives sampled from >=100 positions away.

All args: `--output_csv`, `--triplet_types`, `--min_distance`, `--num_rows`, `--seed`. Run `--help` for details.

#### VBT triplets (Stage 2 — Saiva domain)

```bash
cd data
python vbt_triplets.py                                 # ~630 triplets from 4 strategies
python vbt_triplets.py --strategies A C                # subset of strategies
python vbt_triplets.py --n_negs 5                      # more negatives per sim pair
```

Generates triplets from 168 VBT verses using 4 strategies: (A) direct from similarity pairs, (B) combinatorial within-domain expansion, (C) cross-lingual Sa->En, (D) reverse En->Sa. All hard negatives sourced from Itihasa corpus. Uses union-find over similarity pairs to identify practice domains.

All args: `--output_csv`, `--sa_embedding_path`, `--strategies`, `--n_negs`, `--seed`. Run `--help` for details.

#### Full two-stage data prep workflow

```bash
cd data
# Stage 1: Itihasa
python itihasa_triplets.py                             # -> itihasa_triplets.csv
python nli_preprocess.py --input_csv itihasa_triplets.csv \
  --output_dir ./processed_itihasa/                    # -> processed_itihasa/

# Stage 2: VBT
python vbt_triplets.py                                 # -> vbt_triplets.csv
python nli_preprocess.py --input_csv vbt_triplets.csv \
  --output_dir ./processed_shaiva/                     # -> processed_shaiva/
```

### Training (multi-GPU with DDP via Accelerate)

**Single-stage (Sarvam-1):**
```bash
cd train
./train_sarvam.sh          # multi-GPU DDP training
./train_sarvam_local.sh    # local dev (MPS/CPU, pilot data)
```

**Two-stage (load pre-trained adapter for continued fine-tuning):**
```bash
cd train
./train_sarvam_stage2.sh output/<stage1_timestamp>
```

Stage 2 uses lower learning rate (2e-5 vs 5e-5) and fewer steps (500 vs 1000) to preserve stage-1 gains. Pass `--adapter_path` to `train.py` for any custom two-stage workflow.

**Legacy (MiniCPM):**
```bash
cd train
./train.sh                 # multi-GPU DDP training
./train_local.sh           # local dev
```

The train scripts use `accelerate launch` with the DDP config. Output adapters are saved to `train/output/<timestamp>/`.

### Evaluation (MTEB benchmarks)

Both MTEB eval scripts accept CLI args and log metrics to Weights & Biases. Training and eval always run on remote GPU instances — `train/output/` is empty locally by design.

```bash
cd eval/mteb
python minicpm_sts_eval.py \
  --adapter_path ../../train/output/<timestamp> \
  --wandb_name sts-eval

python minicpm_retrieval_eval.py \
  --adapter_path ../../train/output/<timestamp> \
  --wandb_name retrieval-eval
```

All args (`--model_path`, `--adapter_path`, `--wandb_project`, `--wandb_name`) have sensible defaults (Sarvam-1 base model). Run `--help` for details.

Results saved to `eval/mteb/results/minicpm/`.

### Evaluation (Sanskrit STS)

Custom Sanskrit STS evaluation using VBT (Vijnanabhairava Tantra) benchmark pairs:

```bash
# One-time: generate eval pairs JSON from VBT corpus
cd eval && python vbt_to_json.py

# Run evaluation
python sanskrit_sts_eval.py \
  --eval_data vbt_eval_pairs.json \
  --adapter_path ../train/output/<timestamp> \
  --wandb_name sanskrit-sts
```

Reports 4 metrics: mean similarity (similar pairs), mean similarity (dissimilar pairs), discrimination (delta), and AUC-ROC. All logged to W&B.

### Evaluation (Baseline comparison on Trika data)

Benchmarks off-the-shelf embedding models (LaBSE, E5-multilingual, BGE-M3, Vyakyarth) against Sarvam-1 (base and fine-tuned) on held-out Trika eval data (Śiva Sūtra + Spanda Kārikā). Reports cross-lingual retrieval (MRR, R@k), STS correlation (Spearman ρ), triplet discrimination, and anisotropy.

```bash
cd eval
python baseline_comparison.py                                        # all models
python baseline_comparison.py --models labse e5 sarvam_ft            # subset
python baseline_comparison.py --adapter_path ../train/output/<timestamp>  # with FT model
python baseline_comparison.py --no_wandb                             # skip W&B logging
```

Eval data is in `eval/trika_eval_data.py` — a shared module with verse corpora, STS pairs, and triplets extracted from `sn_model_playground.ipynb`.

All args (`--models`, `--model_path`, `--adapter_path`, `--wandb_project`, `--wandb_name`, `--device`, `--no_wandb`). Run `--help` for details.

## Architecture

### Training pipeline (`train/`)

- **`train.py`** — Entry point. Parses `ModelArguments` (includes `adapter_path` for two-stage training), `DataArguments`, and custom `TrainingArguments` (adds `temperature` param) via HfArgumentParser. Loads a causal LM, wraps it with LoRA via PEFT (or loads a pre-trained adapter), loads preprocessed dataset, and runs `ContrastiveTrainer`.
- **`contrastive_trainer.py`** — Subclass of HuggingFace `Trainer`. Overrides `compute_loss` to encode three inputs (anchor/`sent0`, positive/`sent1`, hard negative/`hard_neg`) by extracting the last hidden state at the final token position, then passes embeddings to InfoNCE loss.
- **`loss.py`** — `InfoNCE` module. Normalizes embeddings, uses `AllGather` across GPUs for global batch negatives, computes cosine similarity logits, and applies cross-entropy with temperature scaling.
- **`utils.py`** — Custom `AllGather` autograd function for gradient-enabled all-gather across distributed processes.

### Data (`data/`)

- **`nli_preprocess.py`** — Tokenizes triplets (sent0, sent1, hard_neg) with a configurable tokenizer (default: Sarvam-1), padding to configurable max_length (default: 150). Saves as HuggingFace dataset. Accepts `--tokenizer_path`, `--max_length`, `--input_csv`, `--output_dir`, `--num_rows`.
- **`itihasa_triplets.py`** — Downloads `rahular/itihasa` (93K Sanskrit-English parallel pairs). Generates cross-lingual `(sn, en, distant_en)` and monolingual Sanskrit `(sn[i], sn[i+1], distant_sn)` triplets with positional distance-based hard negatives.
- **`vbt_triplets.py`** — Generates ~630 triplets from 168 VBT verses via 4 strategies (sim pairs, within-domain expansion, cross-lingual, reverse cross-lingual). Uses union-find for domain detection. Hard negatives from Itihasa corpus.

### Evaluation (`eval/mteb/`)

- **`model/causal_lm.py`** — `CausalLMEncoder` wrapper class with `encode()` method for MTEB compatibility. Extracts embeddings from the last hidden state of the final token. Optionally loads a LoRA adapter. Works with any LlamaForCausalLM-compatible model.
- **`minicpm_sts_eval.py`** / **`minicpm_retrieval_eval.py`** — Accept CLI args (`--model_path`, `--adapter_path`, `--wandb_project`, `--wandb_name`) and log per-task metrics + summary tables to W&B.

### Evaluation (`eval/`)

- **`sanskrit_sts_eval.py`** — Custom Sanskrit STS evaluation. Encodes VBT verse pairs, computes cosine similarity, reports discrimination and AUC-ROC. Logs to W&B.
- **`vbt_to_json.py`** — One-time utility to convert VBT corpus similarity/dissimilarity pairs to JSON eval format.
- **`trika_eval_data.py`** — Shared module with held-out Trika evaluation data (Śiva Sūtra + Spanda Kārikā verse corpora, 21 STS pairs, 16 triplets). Used by `baseline_comparison.py`.
- **`baseline_comparison.py`** — Benchmarks off-the-shelf embedding models (LaBSE, E5, BGE-M3, Vyakyarth) against Sarvam-1 on Trika eval data. Reports cross-lingual retrieval, STS correlation, triplet discrimination, and anisotropy. Logs to W&B.

### Coding conventions

- **Always use `tqdm`** for loops that process more than a trivial number of items (encoding sentences, iterating over dataset rows, etc.). Include a `desc` label and `unit`. Log a summary line with total count, elapsed time, and throughput after completion.

### Key design decisions

- Embeddings are extracted from the **last token** (EOS) of the **last hidden layer** — the tokenizer must have `add_eos_token: true` set (enforced programmatically in all code paths).
- LoRA targets `q_proj` and `v_proj` by default (rank 8, alpha 32, dropout 0.1). Configurable via `--lora_target_modules`.
- **Two-stage training**: pass `--adapter_path` to `train.py` to load a pre-trained LoRA adapter and continue fine-tuning. The adapter is loaded with `is_trainable=True` via `PeftModel.from_pretrained()`.
- In-batch negatives are combined with explicit hard negatives in the InfoNCE loss. `AllGather` enables using negatives across all GPUs for a larger effective batch.
- Pretrained models are expected at `pretrained/` (gitignored). Trained adapters go to `train/output/` (also gitignored).

## Remote Training Practices

When running training or evaluation on a remote GPU instance (Lambda Cloud, etc.), always follow these conventions:

0. **Install GPU drivers first** — On a fresh Lambda Cloud instance, `nvidia-smi` may not work. Before anything else, run `/setup-gpu` (or follow the steps in `.claude/skills/setup-gpu/SKILL.md`) to install the NVIDIA driver. Do not proceed with environment setup or training until `nvidia-smi` shows the GPU correctly.

1. **Always use `tmux`** — Every SSH session to a remote machine must start inside a `tmux` session so that long-running jobs survive disconnects.
   ```bash
   tmux new -s training
   # or reattach: tmux attach -t training
   ```

2. **Always use `tee` for log capture** — Pipe stdout/stderr to a timestamped log file while still displaying output in the terminal.
   ```bash
   ./train.sh 2>&1 | tee train_$(date +%Y%m%d%H%M%S).log
   ```

3. **Always send a push notification on completion** — Append an `ntfy.sh` curl after every long-running command so the user gets notified on their iPhone. Use `set -o pipefail` so `$?` reflects the training script's exit code, not `tee`'s.
   ```bash
   set -o pipefail; ./train.sh 2>&1 | tee train.log; \
     curl -d "Training finished (exit code: $?)" ntfy.sh/LM-STS-CFT
   ```

When composing remote commands, combine all three:
```bash
tmux new -s training
# then inside tmux:
set -o pipefail; ./train.sh 2>&1 | tee train_$(date +%Y%m%d%H%M%S).log; \
  curl -d "Training finished (exit code: $?)" ntfy.sh/LM-STS-CFT
```

4. **JupyterLab for data tasks** — Run JupyterLab on the remote in a dedicated tmux session, then SSH-tunnel the port locally.

   On the remote:
   ```bash
   tmux new -s jupyter
   cd ~/Language-Model-STS-CFT && source .venv/bin/activate
   jupyter lab --no-browser --port 8888 --ip 127.0.0.1
   ```

   On the local machine:
   ```bash
   ssh -fNL 8888:127.0.0.1:8888 lambda
   ```

   Then open the `http://127.0.0.1:8888/lab?token=...` URL printed in the remote tmux session.

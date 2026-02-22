# Architecture: Contrastive Fine-Tuning for Sanskrit Semantic Search

This document describes the system architecture, design rationale, and engineering trade-offs of the LM-STS-CFT project — a pipeline for repurposing causal language models as sentence embedding models via contrastive learning.

---

## 1. Problem Statement

We want a sentence embedding model for **Sanskrit semantic textual similarity (STS)** and **cross-lingual Sanskrit-English retrieval**. The constraints:

- No pre-trained Sanskrit sentence embedding model exists
- Available Sanskrit-capable LMs are causal (decoder-only), not encoder models
- The target domain (Saiva tantric literature) has ~200 annotated verse pairs, far too few to train from scratch
- Inference must work on a single GPU; training budget is ~10 minutes on 1x A100

The solution: take an existing Sanskrit-capable causal LM (Sarvam-1, 2B parameters), freeze most of its weights, and teach it to produce discriminative sentence embeddings via LoRA adaptation and InfoNCE contrastive loss, using a two-stage curriculum.

---

## 2. System Overview

```
                         ┌─────────────────────────────────────────┐
                         │              Data Pipeline              │
                         ├─────────────────────────────────────────┤
                         │  rahular/itihasa ─--> itihasa_triplets  │
                         │  (93K Sa-En pairs)    (186K triplets)   │
                         │                                         │
                         │  VBT corpus ─--> vbt_triplets           │
                         │  (168 verses)    (411 triplets)         │
                         │                                         │
                         │  nli_preprocess.py ─--> tokenized HF DS │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │           Training Pipeline             │
                         ├─────────────────────────────────────────┤
                         │  Sarvam-1 (2B, frozen)                  │
                         │       + LoRA (q_proj, v_proj)           │
                         │                                         │
                         │  Stage 1: Itihasa (1000 steps, lr=5e-5) │
                         │       ↓ adapter checkpoint               │
                         │  Stage 2: VBT (500 steps, lr=2e-5)     │
                         │                                         │
                         │  Loss: InfoNCE(τ=0.05) + AllGather      │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │          Inference / Eval               │
                         ├─────────────────────────────────────────┤
                         │  CausalLMEncoder                        │
                         │    base model + LoRA adapter             │
                         │    embedding = last_hidden[-1, -1, :]   │
                         │                                         │
                         │  MTEB STS/Retrieval benchmarks           │
                         │  Sanskrit STS eval (VBT pairs)          │
                         │  Trika model playground (SS + SK)       │
                         └─────────────────────────────────────────┘
```

---

## 3. Backbone Selection

### Why Sarvam-1

| Criterion | MiniCPM-2B | Sarvam-1 | Winner |
|-----------|------------|----------|--------|
| Sanskrit tokenization | Byte-fallback, ~12× fertility | Devanagari subwords, ~3.9× fertility | Sarvam-1 |
| Parameter count | 2.7B | 2B | Sarvam-1 |
| Architecture | MiniCPM (modified LLaMA) | LLaMA-2 | Tie |
| Pre-training data | Primarily Chinese + English | 22 Indic languages including Sanskrit | Sarvam-1 |

**Tokenization fertility is the decisive factor.** A model that tokenizes "चैतन्यमात्मा" (consciousness-is-Self) into 3 meaningful Devanagari subwords produces far richer per-token representations than one that fragments it into 12 byte-level tokens. Since our embedding strategy extracts a single vector from the final token, every token position matters — fewer tokens means each one must carry more semantic weight, but also means the model has actually *learned* subword-level Sanskrit morphology during pre-training rather than treating it as opaque bytes.

### Why a causal LM at all

Encoder models (BERT-style) are the conventional choice for sentence embeddings. But no encoder model exists with meaningful Sanskrit pre-training. Training one from scratch requires billions of tokens of Sanskrit text and months of compute — neither is available.

Causal LMs, by contrast, already exist for Indic languages. The key insight is that a decoder's last-token hidden state, after seeing the full input sequence via causal attention, functions as a summary embedding of the entire input. With the right training signal (contrastive loss), this representation can be steered into a discriminative embedding space without modifying the model's autoregressive capability.

---

## 4. Embedding Extraction

### Strategy: EOS token, last hidden layer

```python
def encode(self, model, x):
    out = model(**x, output_hidden_states=True).hidden_states[-1][:, -1, :]
    return out
```

The embedding for a sentence is the **hidden state at the final token position** (which is always EOS, enforced by `add_eos_token=True` in the tokenizer) from the **last transformer layer**.

### Rationale

**Why the last token?** In a causal (left-to-right) model, only the final token has attended to every preceding token. Earlier positions have progressively narrower context windows. The last token is the natural "summary" position — it's the same position the model uses to predict the next token during pre-training, so it already encodes a compressed representation of the full sequence.

**Why require EOS?** Without an explicit EOS token, the "last position" is the final content token, which varies unpredictably across inputs (sometimes a noun, sometimes a particle, sometimes punctuation). The EOS token provides a consistent extraction point — a dedicated "summarize everything before me" position. This is enforced programmatically:

```python
self.tokenizer.add_eos_token = True  # set in every code path
```

**Why the last layer?** Lower layers encode increasingly local/syntactic features; the final layer's representations are the most semantically abstract. For sentence-level similarity, we want the most abstract representation available.

**Why not mean pooling?** Mean pooling averages across all token positions, which dilutes the signal with padding tokens and early-layer positional artifacts. For causal models specifically, earlier tokens have seen less context, so averaging them in diminishes the quality of the representation. EOS-token extraction concentrates the signal at the single highest-context point.

**Known limitation:** Ultra-short inputs (2-3 tokens + EOS) produce embeddings dominated by the few tokens available, leading to a "hub" effect where short sūtras with shared morphemes cluster together regardless of semantic content. This is an inherent trade-off of single-vector extraction.

---

## 5. Parameter-Efficient Adaptation (LoRA)

### Configuration

```python
LoraConfig(
    init_lora_weights="gaussian",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    inference_mode=False,
)
```

### Why LoRA

Full fine-tuning of a 2B model for 186K triplets risks catastrophic forgetting — the model's pre-trained Sanskrit language understanding would degrade, defeating the purpose. LoRA constrains adaptation to low-rank perturbations of the attention weights, preserving the bulk of pre-trained knowledge while allowing the embedding geometry to reorganize.

**Trainable parameters**: ~0.05% of total. The base model's 2B parameters remain frozen; only the LoRA adapters (~1M parameters) are updated.

### Why q_proj and v_proj

The query and value projections in self-attention directly control *what the model attends to* (queries) and *what information flows forward* (values). Adapting these is sufficient to redirect the model's attention patterns toward semantically discriminative features.

The key projection (`k_proj`) is deliberately excluded. Keys determine which tokens are *similar to* the query — modifying them alongside queries would create conflicting learning signals. Leaving keys frozen means the adapted model still uses the pre-trained notion of "which tokens relate to each other" while changing what it *does* with that relation.

Output projections (`o_proj`) and MLP layers are excluded to minimize parameter count. The empirical result (MRR 0.76, 100% cross-lingual triplet accuracy) validates that q/v adaptation alone is sufficient.

### Why Gaussian initialization

LoRA weights are initialized from a Gaussian distribution (rather than zeros) to break symmetry across the rank-8 dimensions from the first gradient step. Zero initialization would require multiple steps for the optimizer to differentiate the rank dimensions, wasting early training steps.

### Why rank 8, alpha 32

The effective learning rate scaling for LoRA is `alpha / r = 32 / 8 = 4×`. This means LoRA updates have 4× the magnitude of standard gradient updates, which is appropriate for our setting: we want aggressive reorganization of the embedding space (the anisotropy cure requires moving from mean cosine 0.96 to 0.48) within a limited training budget (1000 + 500 steps).

Higher rank (16, 32) would add capacity but also add parameters that may overfit on the 411-triplet Stage 2. Rank 8 is a pragmatic ceiling for our smallest training set.

---

## 6. Contrastive Loss (InfoNCE)

### Implementation

```python
class InfoNCE(nn.Module):
    def forward(self, query, pos, neg):
        # Normalize to unit sphere
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        # All-gather across GPUs for larger negative pool
        all_pos = AllGather.apply(pos)   # [B*world_size, E]
        all_neg = AllGather.apply(neg)   # [B*world_size, E]

        # Cosine similarity logits
        logits_pos = query @ all_pos.T   # [B, B*world_size]
        logits_neg = query @ all_neg.T   # [B, B*world_size]
        logits = torch.cat((logits_pos, logits_neg), dim=1)

        # Cross-entropy: each query's positive is at its diagonal index
        labels = torch.arange(len(query)) + local_batch_size * rank
        loss = F.cross_entropy(logits / temperature, labels)
        return loss
```

### Why InfoNCE

InfoNCE (Oord et al., 2018) is the standard contrastive objective for representation learning. It treats the task as (B + B)-way classification: for each query, identify its positive among all positives + all negatives in the batch.

**Why not triplet loss?** Triplet loss operates on individual (anchor, positive, negative) tuples and is sensitive to margin hyperparameters. InfoNCE uses the full batch as the negative set, creating a much richer learning signal per step. With batch size 7, each query contends against 7 positives (one correct) + 7 hard negatives = 14 candidates per step. Triplet loss would only see 1 positive + 1 negative.

**Why not cosine similarity loss?** SimCSE-style cosine losses are effective but require careful negative sampling strategies. InfoNCE naturally incorporates in-batch negatives (other samples' positives become implicit negatives), which increases the effective negative pool without additional sampling.

### In-batch negatives + explicit hard negatives

The logit matrix concatenates `logits_pos` and `logits_neg`:

```
logits = [ cos(q_i, pos_0) ... cos(q_i, pos_B) | cos(q_i, neg_0) ... cos(q_i, neg_B) ]
```

This means each query sees:
- **1 true positive** (its own `pos_i`, at diagonal position `i`)
- **B-1 in-batch positives** (other queries' positives — these are implicit negatives)
- **B explicit hard negatives** (from the `hard_neg` column)

The hard negatives are critical for learning fine-grained distinctions. In-batch negatives alone tend to be "easy" (random sentences from the training corpus), while the explicitly mined hard negatives (distant positions in Itihāsa, or cross-domain verses) force the model to learn subtler discriminations.

### Temperature (τ = 0.05)

The temperature scales logits before softmax: `logits / τ`. Lower temperature sharpens the distribution, making the loss more sensitive to small cosine differences.

τ = 0.05 is aggressive (typical values range 0.05–0.1). This choice reflects our goal: we need the model to *strongly* differentiate between similar and dissimilar Sanskrit verses in a space where the base model sees everything as 0.95+ cosine. A sharp temperature punishes the model severely for ranking any negative above the positive, even by a small margin.

### AllGather for distributed training

```python
class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(tensor):
        all_tensor = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(all_tensor, tensor)
        return torch.cat(all_tensor, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output[start:end]  # slice for local rank
```

In multi-GPU DDP training, each GPU sees a different batch slice. Without AllGather, each GPU's InfoNCE loss would only use local negatives (batch_size per GPU). AllGather collects embeddings across all GPUs, creating a global negative pool of `batch_size × num_gpus`, which improves contrastive learning efficiency.

The custom `autograd.Function` ensures gradients flow correctly through the all-gather operation — the backward pass uses `all_reduce(SUM)` to aggregate gradients, then slices out the local rank's portion.

**Note:** The current deployment uses a single A100, so AllGather is effectively a no-op (guarded by `dist.is_initialized()`). The implementation exists for multi-GPU scaling.

---

## 7. Two-Stage Training Curriculum

### Motivation

The target task (Saiva tantric verse similarity) has only 411 training triplets — far too few to teach a model Sanskrit sentence understanding from scratch. But 93K parallel Itihāsa pairs provide abundant general Sanskrit-English alignment signal. The two-stage curriculum exploits this asymmetry:

```
Stage 1: General Sanskrit alignment
  Data:  186K Itihāsa triplets (cross-lingual + monolingual)
  Goal:  Cure anisotropy, learn Sa↔En embedding alignment
  Steps: 1000, lr=5e-5, warmup=100

          │
          ▼ save adapter checkpoint

Stage 2: Domain-specific fine-tuning
  Data:  411 VBT triplets (4 strategies)
  Goal:  Sharpen discrimination on tantric/philosophical text
  Steps: 500, lr=2e-5, warmup=50
```

### Stage 1: Itihāsa (general Sanskrit)

**Data**: `rahular/itihasa` from HuggingFace — 93K Sanskrit-English parallel pairs from the Rāmāyaṇa and Mahābhārata epics.

**Two triplet types:**
- *Cross-lingual*: `(Sanskrit_i, English_i, English_j)` where `|i-j| >= 100` positions
- *Monolingual Sanskrit*: `(Sanskrit_i, Sanskrit_{i+1}, Sanskrit_j)` where `|i-j| >= 100`

The ≥100 position gap for negatives ensures thematic distance without needing chapter metadata (which the dataset lacks). Adjacent verses in epic narrative are narratively continuous and therefore thematically related — a natural source of positive pairs.

**What Stage 1 achieves:** The model learns (a) that Sanskrit and English translations of the same verse should be close, (b) that narratively distant verses should be far apart, and (c) a general reorganization of the embedding space from the anisotropic cone (all cosines ~0.96) to a more uniform distribution.

### Stage 2: VBT (Saiva domain)

**Data**: 411 triplets from 168 Vijñānabhairava Tantra verses, generated by 4 strategies:

| Strategy | Count | Description |
|----------|-------|-------------|
| A: Sim pairs × n_negs | 177 | Each of ~48 expert-annotated similarity pairs × 3 Itihāsa negatives |
| B: Within-domain | 28 | Combinatorial pairs within union-find domains, deduped against A |
| C: Cross-lingual Sa→En | 168 | `(VBT_verse, VBT_translation, Itihāsa_en_negative)` |
| D: Cross-lingual En→Sa | 168 | `(VBT_translation, VBT_verse, Itihāsa_sa_negative)` |

**Why Itihāsa negatives, not cross-domain VBT?** All 168 VBT verses describe dhāraṇā (concentration) practices. Even verses in different "domains" (breath vs. sound vs. void meditation) share heavy thematic overlap — they all discuss states of consciousness, methods of meditation, and the nature of awareness. Using cross-domain VBT verses as negatives would teach the model that "breath meditation" and "sound meditation" are unrelated, which contradicts the actual semantics. Itihāsa epic narrative (battles, genealogies, court intrigue) is genuinely distinct content that provides clean negative signal.

**Why lower learning rate?** Stage 2 uses lr=2e-5 (vs. Stage 1's 5e-5) to avoid catastrophic forgetting of Stage 1 gains. The adapter is loaded with `PeftModel.from_pretrained(model, path, is_trainable=True)` and fine-tuning continues from the Stage 1 checkpoint.

### Domain map via union-find

The VBT similarity pairs implicitly define practice domains (groups of verses that share a meditation technique). Rather than hardcoding these, we discover them via union-find:

```python
def build_domain_map(sim_pairs, corpus):
    """Two verses sharing a similarity pair belong to the same domain.
    Connected components = practice domains."""
    parent = list(range(len(corpus)))
    # union-find over similarity pairs
    for text_a, text_b in sim_pairs:
        union(idx_a, idx_b)
    return {idx: find(idx) for idx in range(len(corpus))}
```

This produces 29 small domains (mostly pairs of 2-3 verses) rather than the expected 8-10 larger thematic groups — indicating that the expert annotations are sparse and don't form large connected components. Strategy B (within-domain expansion) thus generates fewer triplets than originally estimated (28 vs. 150-200), but the cross-lingual strategies C and D provide the bulk of the training signal.

---

## 8. Data Pipeline

### Preprocessing

```
raw CSV (sent0, sent1, hard_neg)
    │
    ▼  nli_preprocess.py
tokenized HF Dataset
    columns: sent0_input_ids, sent0_attention_mask,
             sent1_input_ids, sent1_attention_mask,
             hard_neg_input_ids, hard_neg_attention_mask
    format: torch tensors, max_length=150, padding='max_length'
```

**Why max_length=150?** Itihāsa verses and VBT ślokas rarely exceed 100 tokens with the Sarvam-1 tokenizer. 150 provides headroom without excessive padding waste. Longer sequences would increase memory consumption linearly without adding signal for this corpus.

**Why padding to max_length?** The HuggingFace Trainer's default collator expects fixed-length tensors. Dynamic padding would be more memory-efficient but requires a custom data collator — unnecessary complexity for our dataset sizes.

### ContrastiveTrainer data handling

The `ContrastiveTrainer` overrides `compute_loss` to unpack the 6 tensor columns into 3 `(input_ids, attention_mask)` pairs:

```python
def compute_loss(self, model, inputs, return_outputs=False):
    sent0 = {'input_ids': inputs['sent0_input_ids'],
             'attention_mask': inputs['sent0_attention_mask']}
    # ... encode all three, compute InfoNCE
```

This is simpler than a custom data collator and keeps the Trainer's built-in features (gradient accumulation, mixed precision, checkpointing) working without modification.

---

## 9. Inference Architecture

### CausalLMEncoder

```python
class CausalLMEncoder:
    def __init__(self, model_path, adapter_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cuda')
        if adapter_path:
            self.model.load_adapter(adapter_path)

    def encode(self, sentences):
        out = []
        for s in sentences:
            inputs = self.tokenizer(s, return_tensors="pt").to("cuda")
            hidden = self.model(**inputs, output_hidden_states=True)
            emb = hidden.hidden_states[-1][0, -1, :]  # last layer, last token
            out.append(emb.float().cpu().numpy())
        return np.array(out)
```

**Why sequential encoding?** Batched encoding with padding introduces padding tokens that shift the "last token" position. The simple sequential approach avoids this — each sentence is encoded individually with no padding. At ~30 sentences/sec on A100 (measured), this is fast enough for evaluation-scale corpora (tens to hundreds of sentences). Production-scale retrieval would need batched encoding with proper attention masking.

**Why float32 output?** The model runs in bfloat16 for speed, but the final embedding is cast to float32 for cosine similarity computation. Bfloat16 has limited mantissa precision (7 bits vs. 23 for float32), which can introduce noise in similarity scores — especially when computing small differences between high-similarity pairs.

**Why `load_adapter` not `PeftModel.from_pretrained`?** For inference, `load_adapter()` is simpler — it attaches the LoRA weights to the existing model without wrapping it in a PeftModel. This is sufficient for forward-pass-only usage and avoids the `is_trainable` flag complexity.

### CUDA cache management

```python
if (i + 1) % CACHE_CLEAR_INTERVAL == 0:
    torch.cuda.empty_cache()
```

Every 100 sentences, the CUDA cache is explicitly cleared. Without this, encoding hundreds of variable-length sentences accumulates fragmented memory allocations that can trigger OOM on 40GB GPUs despite the actual working set being small.

---

## 10. Evaluation Design

### Three evaluation tiers

| Tier | Tool | What it measures | Data |
|------|------|------------------|------|
| **MTEB benchmarks** | `minicpm_sts_eval.py`, `minicpm_retrieval_eval.py` | Standard English STS/retrieval | MTEB benchmark suite |
| **Sanskrit STS** | `sanskrit_sts_eval.py` | Discrimination + AUC-ROC on VBT pairs | `vbt_eval_pairs.json` (similarity + dissimilarity pairs) |
| **Model playground** | `sn_model_playground.ipynb` | 12 qualitative + quantitative analyses on held-out Trika texts | Siva Sutra + Spanda Karika (not in training set) |

### Why held-out Trika texts for the playground

The VBT eval pairs test in-distribution performance. The Siva Sutra and Spanda Karika provide a stronger test: they are from the same *philosophical tradition* (Kashmir Saivism / Trika) but are different texts by different authors in different literary forms (sūtra vs. kārikā). Success on these indicates the model has learned generalizable Sanskrit semantic representations, not just memorized VBT-specific patterns.

### Metrics reported

- **STS correlation** (Spearman ρ): Rank correlation between model cosine and human similarity judgments
- **Cross-lingual retrieval** (MRR, R@1, R@3, R@5): Given a verse in one language, rank-retrieve its translation from the other
- **Triplet accuracy**: Does cos(anchor, positive) > cos(anchor, negative)?
- **Uniformity** (Wang & Isola 2020): Distribution quality on the unit hypersphere
- **Discrimination** (δ): Mean sim(similar pairs) − mean sim(dissimilar pairs)
- **AUC-ROC**: Binary classification of similar vs. dissimilar pairs by cosine threshold
- **Embedding drift**: cos(base_embedding, ft_embedding) measures how far training moved representations

---

## 11. Design Decisions Not Taken

### Pooling alternatives (rejected)

- **Mean pooling over all tokens**: Dilutes signal with padding and low-context early tokens
- **Mean of last 4 layers**: Adds complexity; the last layer already captures the most abstract semantics
- **Attention-weighted pooling**: Requires an additional learned pooling head, adding trainable parameters and potential overfitting

### Larger LoRA rank (deferred)

Rank 16 or 32 would provide more adaptation capacity but risks overfitting on the 411-triplet Stage 2 dataset. The rank-8 configuration achieves 100% cross-lingual triplet accuracy, suggesting the capacity ceiling has not been reached on the *easy* tasks but is adequate for the current data scale.

### Multi-task training (deferred)

Joint training on cross-lingual retrieval + monolingual STS + classification could improve monolingual performance. However, this requires task-specific loss weighting and a larger annotated dataset. The current pipeline prioritizes simplicity — a single InfoNCE loss on triplet data.

### Contrastive pre-training of the full model (rejected)

Training all 2B parameters with contrastive loss would maximize adaptation but requires 10-100× more data and compute to avoid catastrophic forgetting. LoRA achieves the embedding reorganization we need (mean pairwise cosine 0.96 → 0.48) while preserving the pre-trained language model.

---

## 12. File Map

```
Language-Model-STS-CFT/
├── data/
│   ├── download_sarvam.sh          # Download Sarvam-1, set add_eos_token
│   ├── download_nli.sh             # Download English NLI triplets
│   ├── itihasa_triplets.py         # Stage 1 data: 186K Sa-En triplets from Itihasa
│   ├── vbt_triplets.py             # Stage 2 data: 411 VBT triplets (4 strategies)
│   └── nli_preprocess.py           # Tokenize triplet CSV -> HF Dataset
│
├── train/
│   ├── train.py                    # Entry point: args, model loading, LoRA setup
│   ├── contrastive_trainer.py      # HF Trainer subclass: encode + compute_loss
│   ├── loss.py                     # InfoNCE with AllGather
│   ├── utils.py                    # AllGather autograd function for DDP
│   ├── configs/
│   │   ├── ddp_config.yaml         # Accelerate config for multi-GPU DDP
│   │   └── local_config.yaml       # Accelerate config for local dev (MPS/CPU)
│   ├── train_sarvam.sh             # Stage 1 launch script
│   ├── train_sarvam_stage2.sh      # Stage 2 launch script (takes Stage 1 adapter path)
│   ├── train_sarvam_local.sh       # Local dev script
│   ├── train.sh                    # Legacy MiniCPM launch script
│   └── train_local.sh              # Legacy MiniCPM local script
│
├── eval/
│   ├── sanskrit_sts_eval.py        # VBT similarity/dissimilarity eval + W&B logging
│   ├── vbt_to_json.py              # Convert VBT corpus to eval JSON
│   ├── sn_model_playground.ipynb   # 12-section held-out Trika evaluation notebook
│   ├── evaluation_review.md        # Full evaluation report with results
│   └── mteb/
│       ├── model/
│       │   └── causal_lm.py        # CausalLMEncoder: embedding extraction wrapper
│       ├── minicpm_sts_eval.py     # MTEB STS benchmark runner
│       └── minicpm_retrieval_eval.py  # MTEB retrieval benchmark runner
│
├── pretrained/                     # (gitignored) Downloaded model weights
├── pyproject.toml                  # Dependencies: torch 2.2, transformers 4.40, peft 0.10
└── architecture.md                 # This document
```

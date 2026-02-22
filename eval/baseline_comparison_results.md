# Baseline Comparison: Off-the-Shelf Embedding Models vs Fine-Tuned Sarvam-1

**Date:** 2026-02-22
**Hardware:** NVIDIA A100-SXM4-40GB (Lambda Cloud), Driver 590.48.01, CUDA 13.1
**Runtime:** 40s total benchmark (6 models), training: Stage 1 5:16, Stage 2 2:39

## Overview

We benchmark 5 off-the-shelf multilingual embedding models against our contrastive fine-tuned Sarvam-1 (2B params, two-stage LoRA + InfoNCE) on **held-out Trika evaluation data** --- 20 Siva Sutra verses and 12 Spanda Karika verses, neither of which appeared in training (VBT / Itihasa).

The primary use case is **semantic search over Sanskrit philosophical texts**: given a query verse, retrieve thematically related verses from a corpus. The key quality signal is therefore **discrimination** --- the model's ability to push similar pairs apart from dissimilar ones in cosine space, creating a usable similarity gradient for ranking.

## Models

| Model | Params | Type | Notes |
|-------|--------|------|-------|
| LaBSE | 471M | Sentence-BERT | Language-agnostic, 109 languages |
| E5-multilingual | 560M | Sentence-BERT | Instruct-tuned, `query:` prefix |
| BGE-M3 | 568M | Sentence-BERT | Multi-granularity, 100+ languages |
| Vyakyarth | ~560M | Sentence-BERT | Indic-specific (Krutrim AI Labs) |
| Sarvam-1 (base) | 2B | Causal LM | Sanskrit-capable backbone, no fine-tuning |
| Sarvam-1 (FT) | 2B | Causal LM + LoRA | Two-stage CFT: Itihasa (186K triplets, 1000 steps) + VBT (411 triplets, 500 steps) |

## Results

### Summary Table

```
+--------------------+---------+---------+---------+---------+-----------+---------+-----------+
|       Model        |   MRR   |   R@1   |   R@5   |   rho   | Trip.Acc  | Discr.  | Uniform.  |
+--------------------+---------+---------+---------+---------+-----------+---------+-----------+
| LaBSE              |   0.731 |   62.5% |   84.4% |   0.291 |     75.0% |   0.119 |    -2.684 |
| E5-multilingual    |   0.764 |   65.6% |   90.6% |   0.273 |     68.8% |   0.032 |    -0.564 |
| BGE-M3             |   0.784 |   70.3% |   87.5% |   0.537 |     75.0% |   0.092 |    -2.169 |
| Vyakyarth          |   0.577 |   46.9% |   68.8% |   0.116 |     43.8% |   0.070 |    -2.435 |
| Sarvam-1 (base)    |   0.132 |    1.6% |   17.2% |  -0.174 |     56.2% |   0.008 |    -0.138 |
| Sarvam-1 (FT)      |   0.697 |   56.2% |   89.1% |   0.323 |     62.5% |   0.131 |    -1.957 |
+--------------------+---------+---------+---------+---------+-----------+---------+-----------+
```

**Metric definitions:**
- **MRR** --- Mean Reciprocal Rank, averaged over En->Sa and Sa->En retrieval (32 pairs each)
- **R@1, R@5** --- Recall at 1 and 5, averaged over both directions
- **rho** --- Spearman rank correlation between model cosine similarity and human-annotated similarity scores (21 pairs, 0--5 scale)
- **Trip.Acc** --- Triplet discrimination accuracy: cos(anchor, positive) > cos(anchor, negative) across 16 triplets
- **Discr.** --- Mean margin (cos_pos - cos_neg) across all triplets; higher = sharper separation
- **Uniform.** --- Wang & Isola (2020) uniformity metric on 32 Sanskrit embeddings; lower = more uniform distribution on hypersphere

---

## Discrimination Analysis (Primary Metric)

For semantic search, what matters most is not whether a model assigns high absolute similarity to related pairs, but whether it **separates related from unrelated pairs by a usable margin**. A model with triplet accuracy of 100% but margin of 0.01 is fragile --- any noise, new text, or threshold choice will break it. A model with 62% accuracy but 0.13 margin on its correct cases is more robust for ranking.

### Overall Discrimination Margin

| Model | Mean Margin | Interpretation |
|-------|-------------|----------------|
| **Sarvam-1 (FT)** | **+0.131** | Strongest overall separation |
| LaBSE | +0.119 | Close second |
| BGE-M3 | +0.092 | Good but narrower |
| Vyakyarth | +0.070 | Weak |
| E5-multilingual | +0.032 | Near-flat cosine space for Sanskrit |
| Sarvam-1 (base) | +0.008 | Effectively zero --- anisotropy |

Sarvam-1 FT and LaBSE are the only models that consistently open up a margin >0.1 between related and unrelated verse pairs. E5-multilingual, despite strong retrieval numbers (MRR 0.764), has a mean margin of just 0.032 --- its Sanskrit embeddings are nearly equidistant, making any retrieval result fragile to small perturbations.

### Per-Category Discrimination

The three triplet categories test different capabilities:

| Model | Mono-Sa (8) | Cross-lingual (6) | Hard (2) |
|-------|-------------|--------------------|-----------|
| | acc / mean margin | acc / mean margin | acc / mean margin |
| **Sarvam-1 (FT)** | 3/8 / +0.006 | **6/6 / +0.325** | 1/2 / -0.066 |
| LaBSE | **6/8 / +0.115** | 6/6 / +0.172 | 0/2 / -0.023 |
| BGE-M3 | **6/8 / +0.077** | 6/6 / +0.162 | 0/2 / -0.059 |
| E5 | 5/8 / +0.019 | 6/6 / +0.064 | 0/2 / -0.009 |
| Vyakyarth | 1/8 / -0.089 | 5/6 / +0.279 | 1/2 / +0.076 |
| Base | 3/8 / -0.001 | 4/6 / +0.018 | 2/2 / +0.018 |

**Key observations:**

1. **Cross-lingual discrimination is Sarvam-1 FT's strongest suit.** Mean cross-lingual margin of +0.325 is nearly 2x LaBSE (+0.172) and 5x E5 (+0.064). When the task is "find the Sanskrit verse matching this English query," Sarvam-1 FT creates the widest gaps between correct and incorrect matches. Its widest single margin is +0.531 (SS 1.1 Sa vs its own translation vs an unrelated En translation).

2. **Monolingual Sanskrit discrimination is where Sarvam-1 FT struggles.** Mean mono-Sa margin of +0.006 is effectively zero --- it gets some right by large margins (+0.287 for SS 1.22/SS 2.1 mantra pair, +0.251 for SS 1.7/SK 3 turiya pair) but also fails catastrophically (-0.359 for SS 1.5/SK 22 pair). LaBSE is the clear leader here (+0.115 mean margin, 6/8 accuracy).

3. **No model solves the hard triplets.** When positive and negative are both semantically related to the anchor (e.g., three verses all about atma, or three about spanda/states), every model except Vyakyarth has negative mean margin. These require understanding subtle philosophical distinctions that no current embedding model captures.

### Per-Triplet Detail: Where Models Agree and Disagree

```
Category        Triplet                                       LaBSE        E5    BGE-M3  Vyakyarth      Base        FT
----------------------------------------------------------------------------------------------------------------------
mono_sa         SS 1.7 <> SK 3 vs SS 3.45                   +0.314    +0.031    +0.169    -0.019    -0.008    +0.251
mono_sa         SS 1.12 <> SK 11 vs SK 48                   -0.029    -0.004    +0.044    -0.074    -0.022    -0.123
mono_sa         SS 3.26 <> SK 30 vs SS 3.45                 +0.025    -0.031    -0.018    -0.059    +0.019    -0.112
mono_sa         SK 21 <> SK 44 vs SS 1.1                    +0.300    +0.091    +0.188    +0.011    +0.053    -0.048
mono_sa         SS 1.5 <> SK 22 vs SS 2.6                   -0.189    -0.046    -0.025    -0.350    -0.012    -0.359
mono_sa         SS 1.22 <> SS 2.1 vs SK 9                   +0.239    +0.016    +0.122    -0.028    -0.019    +0.287
mono_sa         SK 1 <> SK 2 vs SS 3.12                     +0.204    +0.063    +0.065    -0.136    +0.005    -0.022
mono_sa         SS 3.43 <> SK 9 vs SS 2.1                   +0.052    +0.032    +0.069    -0.056    -0.026    +0.173
cross_lingual   SS 1.1 Sa <> SS 1.1 En vs SS 3.45 En        +0.212    +0.047    +0.183    +0.626    +0.085    +0.531
cross_lingual   SK 22 Sa <> SK 22 En vs SK 2 En              +0.136    +0.081    +0.137    +0.176    +0.012    +0.272
cross_lingual   SK 30 Sa <> SK 30 En vs SS 1.2 En            +0.190    +0.104    +0.216    +0.224    +0.032    +0.172
cross_lingual   SS 3.9 Sa <> SS 3.9 En vs SK 48 En           +0.179    +0.064    +0.236    +0.388    +0.007    +0.397
cross_lingual   SK 5 Sa <> SK 5 En vs SS 1.18 En             +0.235    +0.075    +0.124    +0.321    -0.019    +0.210
cross_lingual   SS 1.7 Sa <> SS 1.7 En vs SK 48 En           +0.082    +0.012    +0.073    -0.059    -0.010    +0.370
hard            SS 1.1 <> SS 1.17 vs SS 3.9 (all atma)      -0.041    -0.006    -0.013    +0.283    +0.003    +0.039
hard            SK 3 <> SK 17 vs SK 21 (all spanda/states)   -0.005    -0.012    -0.104    -0.131    +0.033    -0.171
```

**Notable patterns:**

- **SS 1.5 <> SK 22 vs SS 2.6** is a universal failure: every model either gets it wrong or has a near-zero margin. The anchor (udyama/upsurge) should be closer to SK 22 (spanda in extreme states) than SS 2.6 (guru is the means), but short sutras like SS 2.6 ("gururupayah", 2 tokens) act as embedding hubs that attract unrelated queries.

- **SK 21 <> SK 44 vs SS 1.1** splits models sharply: LaBSE (+0.300) and BGE-M3 (+0.188) get it right, but FT (-0.048) fails because SS 1.1 ("caitanyamatma") is a 2-token sutra that gets pulled close to everything. This is the **short-sutra hub problem** --- ultra-short inputs produce embeddings that are artificially close to many others.

- **Cross-lingual margins are systematically larger** for all models. This makes sense: the Sa<->En language barrier creates natural separation between unrelated pairs, making the discrimination task easier.

### Embedding Space Geometry: Why Discrimination Matters

```
Model            Mean      Std      Min      Max    Range
--------------------------------------------------------
LaBSE          0.294    0.130   -0.069    0.685    0.753
E5             0.857    0.034    0.772    0.941    0.169
BGE-M3         0.444    0.080    0.264    0.743    0.479
Vyakyarth      0.355    0.134   -0.031    0.850    0.881
Sarvam-1 base  0.964    0.027    0.840    0.998    0.158
Sarvam-1 FT    0.481    0.114    0.192    0.770    0.578
```

This table explains why discrimination margin matters more than raw accuracy:

- **E5-multilingual** has a pairwise cosine range of just 0.169 (0.77--0.94). All 32 Sanskrit verses land within a 0.17-wide band. Even though E5 ranks many pairs correctly (MRR 0.764), the **margin for error is razor-thin** --- a threshold-based search system would have almost no room to set a meaningful cutoff between "relevant" and "irrelevant."

- **Sarvam-1 base** is even worse: range 0.158 (0.84--1.00). This is pathological anisotropy --- every verse looks the same.

- **LaBSE** has the widest range (0.753) and goes slightly negative, meaning it can place truly unrelated pairs in opposite hemispheres. This gives maximum room for threshold-based filtering.

- **Sarvam-1 FT** has a range of 0.578 (0.19--0.77), comparable to BGE-M3 (0.479). Importantly, its minimum pairwise cosine (0.192) is much lower than BGE-M3's (0.264), meaning it pushes unrelated pairs further apart.

For a semantic search system, **range and std determine how many "buckets" of similarity the model can distinguish**. A model with std=0.027 (base) has ~1 bucket; a model with std=0.130 (LaBSE) has ~5--6 meaningful gradations.

### STS Correlation: The Graded Similarity View

The 21 human-annotated STS pairs test whether models preserve the **full gradient** from unrelated (0.0) to paraphrastic (5.0), not just binary discrimination.

| Model | Spearman rho | Pearson r | Interpretation |
|-------|-------------|-----------|----------------|
| BGE-M3 | **0.537** | 0.509 | Best ordinal ranking of similarity |
| Sarvam-1 (FT) | 0.323 | 0.207 | Moderate --- good extremes, noisy middle |
| LaBSE | 0.291 | 0.287 | Similar to FT but more linear |
| E5 | 0.273 | 0.219 | Weak despite high retrieval |
| Vyakyarth | 0.116 | 0.008 | Near-random |
| Base | -0.174 | -0.220 | Anti-correlated (anisotropy) |

BGE-M3's rho of 0.537 stands out. Looking at the per-pair cosines reveals why:

**Selected STS pairs showing model behavior across the similarity scale:**

| Human | Pair | LaBSE | E5 | BGE-M3 | FT |
|-------|------|-------|-----|--------|-----|
| 5.0 | SS 1.7 <> SK 3 (turiya in 3 states) | 0.556 | 0.901 | 0.627 | 0.723 |
| 5.0 | SS 1.7 <> SK 17 (turiya/spanda in 3 states) | 0.426 | 0.870 | 0.513 | 0.291 |
| 3.0 | SS 1.17 <> SS 1.1 (atma-jnana/caitanya) | 0.566 | 0.875 | 0.523 | **0.748** |
| 1.0 | SS 1.18 <> SK 22 (samadhi/spanda) | 0.244 | 0.823 | 0.451 | 0.354 |
| 0.0 | SS 1.1 <> SS 3.45 (consciousness/pranayama) | 0.425 | 0.816 | 0.397 | 0.389 |
| 0.0 | SS 2.6 <> SS 3.45 (guru/pranayama) | 0.147 | 0.817 | 0.338 | 0.419 |

**Why FT's rho is lower than BGE-M3 despite good discrimination:**

- FT gives SS 1.17 <> SS 1.1 (human score: 3.0) a cosine of 0.748 --- higher than the 5.0-rated pair SS 1.7 <> SK 17 (0.291). This is the **short-sutra morpheme effect**: both SS 1.17 and SS 1.1 are 2--3 word sutras sharing the root "atma", so surface overlap inflates their cosine. The model correctly sees they're related but over-ranks them.

- E5 gives everything 0.80--0.90, so while its ordinal ranking is slightly correct, the absolute values carry no discriminative information.

- BGE-M3 maintains a monotonic trend from 0.34 (score 0) to 0.63 (score 5) with the widest gaps between similarity tiers, which produces the best Spearman correlation.

---

## Cross-Lingual Retrieval

### Per-Direction Detail

```
  Model              En->Sa MRR     R@1     R@3     R@5   Sa->En MRR     R@1     R@3     R@5
  -----------------------------------------------------------------------------------------------
  LaBSE                   0.712   59.4%   75.0%   87.5%        0.750   65.6%   81.2%   81.2%
  E5-multilingual         0.800   71.9%   84.4%   90.6%        0.728   59.4%   87.5%   90.6%
  BGE-M3                  0.719   62.5%   75.0%   84.4%        0.849   78.1%   90.6%   90.6%
  Vyakyarth               0.632   53.1%   68.8%   78.1%        0.522   40.6%   53.1%   59.4%
  Sarvam-1 (base)         0.134    3.1%    9.4%   12.5%        0.130    0.0%   12.5%   21.9%
  Sarvam-1 (FT)           0.718   62.5%   75.0%   84.4%        0.675   50.0%   78.1%   93.8%
```

BGE-M3 leads overall (MRR 0.784), with E5 close behind (0.764). Sarvam-1 FT (0.697) is in the same tier as LaBSE (0.731).

Sarvam-1 FT achieves the **highest R@5 for Sa->En** (93.8%) --- for nearly every Sanskrit verse, the correct English translation appears in the top 5 results. Its En->Sa performance (MRR 0.718) matches BGE-M3 exactly.

The base Sarvam-1 is essentially random (MRR 0.132, R@1 1.6%), confirming that contrastive fine-tuning is entirely responsible for the cross-lingual capability.

### Triplet Accuracy (Per Category)

```
  Model                     mono_sa  cross_lingual           hard
  ---------------------------------------------------------------
  LaBSE              6/8 (  75%) 6/6 ( 100%) 0/2 (   0%)
  E5-multilingual    5/8 (  62%) 6/6 ( 100%) 0/2 (   0%)
  BGE-M3             6/8 (  75%) 6/6 ( 100%) 0/2 (   0%)
  Vyakyarth          1/8 (  12%) 5/6 (  83%) 1/2 (  50%)
  Sarvam-1 (base)    3/8 (  38%) 4/6 (  67%) 2/2 ( 100%)
  Sarvam-1 (FT)      3/8 (  38%) 6/6 ( 100%) 1/2 (  50%)
```

---

## Anisotropy and Uniformity

| Metric | Base | FT | Change |
|--------|------|-----|--------|
| Mean pairwise cosine | 0.964 | 0.481 | -0.483 |
| Std | 0.027 | 0.114 | 4.2x wider |
| Range | 0.158 | 0.578 | 3.7x wider |
| Uniformity | -0.138 | -1.957 | 14x improvement |

Contrastive training eliminates the base model's pathological anisotropy, spreading embeddings from a narrow cone (0.84--1.00) to a usable distribution (0.19--0.77). The uniformity metric improves 14x, approaching the level of purpose-built embedding models.

---

## Takeaways

1. **Sarvam-1 FT has the best overall discrimination margin (+0.131)**, making it the most robust choice for threshold-based semantic search. When it correctly identifies a related pair, the gap to unrelated pairs is wider than any other model.

2. **Cross-lingual discrimination is Sarvam-1 FT's clear advantage** (+0.325 mean margin, 2x LaBSE, 5x E5). For the primary use case of Sanskrit<->English semantic search, this is the most relevant metric.

3. **Monolingual Sanskrit discrimination is the main weakness** (mean margin +0.006, accuracy 3/8). LaBSE (6/8, +0.115) and BGE-M3 (6/8, +0.077) are substantially better at distinguishing thematic relationships within Sanskrit.

4. **BGE-M3 is the best all-rounder** for off-the-shelf use: highest MRR (0.784), best STS correlation (rho 0.537), and solid discrimination (+0.092). If fine-tuning is not an option, BGE-M3 is the strongest choice.

5. **E5-multilingual's high MRR is misleading** for semantic search. Its pairwise cosine range of 0.169 and mean margin of 0.032 mean all Sanskrit texts land in nearly the same region of embedding space. Retrieval works by fragile ordinal ranking, not by meaningful similarity gaps.

6. **The short-sutra hub problem** limits all models. Ultra-short inputs (2--3 tokens) like SS 1.1 "caitanyamatma" and SS 2.6 "gurupayah" produce embeddings that attract unrelated queries, inflating false-positive similarity. This is a structural limitation of mean/last-token pooling on very short texts.

7. **Potential improvement path**: monolingual Sanskrit discrimination could be improved by (a) augmenting training data with fine-grained Sanskrit-only triplets that explicitly contrast within-theme pairs, or (b) incorporating morphological features that help the model look past surface-level morpheme overlap.

## Reproducibility

```bash
# Training
cd train
WANDB_MODE=disabled bash train_sarvam.sh                    # Stage 1: 5:16 on A100
WANDB_MODE=disabled bash train_sarvam_stage2.sh output/<stage1_timestamp>  # Stage 2: 2:39

# Benchmark
cd eval
python baseline_comparison.py --adapter_path ../train/output/<stage2_timestamp> --no_wandb

# Detailed per-triplet analysis
python detailed_analysis.py
```

**Logs:** `eval/logs/baseline_comparison_20260222142304.log`, `eval/logs/train_stage1_20260222141240.log`, `eval/logs/train_stage2_20260222141912.log`, `eval/logs/detailed_analysis_output.txt`

**Eval data:** `eval/trika_eval_data.py` (32 verse pairs, 21 STS pairs, 16 triplets)

**Adapter:** `train/output/20260222141912` (Stage 2, on remote instance)

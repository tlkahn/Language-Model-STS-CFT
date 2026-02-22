"""
Baseline comparison: benchmark off-the-shelf embedding models against our
fine-tuned Sarvam-1 on held-out Trika eval data (Śiva Sūtra + Spanda Kārikā).

Models: LaBSE, E5-multilingual, BGE-M3, Vyakyarth, Sarvam-1 (base), Sarvam-1 (FT).

Metrics:
  1. Cross-lingual retrieval (MRR, R@1, R@3, R@5)
  2. STS correlation (Spearman ρ, Pearson r)
  3. Triplet discrimination (accuracy, per-category)
  4. Anisotropy (mean pairwise cosine, std, uniformity)
"""

import argparse
import logging
import sys
import time

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from trika_eval_data import get_all_verses, get_sts_pairs, get_triplets

# CausalLMEncoder lives in eval/mteb/model/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "mteb"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_comparison")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "labse": {
        "hf_id": "sentence-transformers/LaBSE",
        "display": "LaBSE",
        "type": "sentence_transformer",
        "prefix": "",
    },
    "e5": {
        "hf_id": "intfloat/multilingual-e5-large",
        "display": "E5-multilingual",
        "type": "sentence_transformer",
        "prefix": "query: ",
    },
    "bge_m3": {
        "hf_id": "BAAI/bge-m3",
        "display": "BGE-M3",
        "type": "sentence_transformer",
        "prefix": "",
    },
    "vyakyarth": {
        "hf_id": "krutrim-ai-labs/Vyakyarth",
        "display": "Vyakyarth",
        "type": "sentence_transformer",
        "prefix": "",
    },
    "sarvam_base": {
        "display": "Sarvam-1 (base)",
        "type": "causal_lm",
    },
    "sarvam_ft": {
        "display": "Sarvam-1 (FT)",
        "type": "causal_lm",
    },
}


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------
class BaselineEncoder:
    """Wrapper for sentence-transformers models to match CausalLMEncoder.encode() interface."""

    def __init__(self, model_id, device, prefix=""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.prefix = prefix

    def encode(self, sentences, **kwargs):
        prefixed = [self.prefix + s for s in sentences]
        return self.model.encode(prefixed, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(query_embs, corpus_embs, ks=(1, 3, 5)):
    """MRR and Recall@k. query_embs[i] should match corpus_embs[i]."""
    n = len(query_embs)
    sims = cosine_similarity(query_embs, corpus_embs)
    ranks = []
    for i in range(n):
        sorted_idx = np.argsort(-sims[i])
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    recalls = {k: float(np.mean(ranks <= k)) for k in ks}
    return mrr, recalls


def compute_sts_correlation(encoder):
    """Spearman ρ and Pearson r on 21 human-annotated STS pairs."""
    pairs = get_sts_pairs()
    human_scores = [p[2] for p in pairs]
    model_scores = []
    for text_a, text_b, _, _ in pairs:
        embs = encoder.encode([text_a, text_b])
        sim = cosine_similarity([embs[0]], [embs[1]])[0, 0]
        model_scores.append(sim)
    rho, _ = spearmanr(human_scores, model_scores)
    r, _ = pearsonr(human_scores, model_scores)
    return float(rho), float(r)


def compute_triplet_accuracy(encoder):
    """Triplet discrimination: cos(anchor, pos) > cos(anchor, neg)?"""
    triplets = get_triplets()
    results = {}
    for label, anchor, positive, negative, category in triplets:
        embs = encoder.encode([anchor, positive, negative])
        pos_sim = cosine_similarity([embs[0]], [embs[1]])[0, 0]
        neg_sim = cosine_similarity([embs[0]], [embs[2]])[0, 0]
        correct = pos_sim > neg_sim
        if category not in results:
            results[category] = {"correct": 0, "total": 0, "margins": []}
        results[category]["correct"] += int(correct)
        results[category]["total"] += 1
        results[category]["margins"].append(float(pos_sim - neg_sim))

    total_correct = sum(v["correct"] for v in results.values())
    total_n = sum(v["total"] for v in results.values())
    overall_acc = total_correct / total_n if total_n > 0 else 0.0
    mean_margin = float(np.mean([m for v in results.values() for m in v["margins"]]))
    return overall_acc, mean_margin, results


def compute_anisotropy(embeddings):
    """Pairwise cosine stats + Wang & Isola (2020) uniformity on Sanskrit embeddings."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)
    triu_idx = np.triu_indices(n, k=1)
    pairwise = sim_matrix[triu_idx]

    # Uniformity: -log E[exp(-2||z_i - z_j||^2)]
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sq_pdist = np.sum((norms[:, None] - norms[None, :]) ** 2, axis=-1)
    triu_dists = sq_pdist[triu_idx]
    uniformity = float(np.log(np.mean(np.exp(-2 * triu_dists))))

    return {
        "mean_cos": float(np.mean(pairwise)),
        "std_cos": float(np.std(pairwise)),
        "uniformity": uniformity,
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_model(encoder, name):
    """Run all 4 metric suites, return results dict."""
    logger.info(f"Evaluating {name}...")
    start = time.time()
    results = {"model": name}

    # 1. Cross-lingual retrieval
    sa_ids, sa_texts, en_ids, en_texts = get_all_verses()
    logger.info(f"  Encoding {len(sa_texts)} Sa + {len(en_texts)} En verses...")
    sa_embs = np.array(encoder.encode(sa_texts))
    en_embs = np.array(encoder.encode(en_texts))

    mrr_en2sa, recalls_en2sa = compute_retrieval_metrics(en_embs, sa_embs)
    mrr_sa2en, recalls_sa2en = compute_retrieval_metrics(sa_embs, en_embs)

    results["mrr_en2sa"] = mrr_en2sa
    results["r1_en2sa"] = recalls_en2sa[1]
    results["r3_en2sa"] = recalls_en2sa[3]
    results["r5_en2sa"] = recalls_en2sa[5]
    results["mrr_sa2en"] = mrr_sa2en
    results["r1_sa2en"] = recalls_sa2en[1]
    results["r3_sa2en"] = recalls_sa2en[3]
    results["r5_sa2en"] = recalls_sa2en[5]
    results["mrr_avg"] = (mrr_en2sa + mrr_sa2en) / 2
    results["r1_avg"] = (recalls_en2sa[1] + recalls_sa2en[1]) / 2

    # 2. STS correlation
    logger.info("  Computing STS correlation...")
    rho, r = compute_sts_correlation(encoder)
    results["spearman"] = rho
    results["pearson"] = r

    # 3. Triplet discrimination
    logger.info("  Computing triplet accuracy...")
    trip_acc, trip_margin, trip_by_cat = compute_triplet_accuracy(encoder)
    results["triplet_acc"] = trip_acc
    results["triplet_margin"] = trip_margin
    results["triplet_by_cat"] = trip_by_cat

    # 4. Anisotropy
    logger.info("  Computing anisotropy...")
    aniso = compute_anisotropy(sa_embs)
    results["mean_cos"] = aniso["mean_cos"]
    results["std_cos"] = aniso["std_cos"]
    results["uniformity"] = aniso["uniformity"]

    elapsed = time.time() - start
    results["eval_time"] = elapsed
    logger.info(f"  Done in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_comparison_table(all_results):
    """Print formatted comparison table to stdout."""
    # Header
    cols = [
        ("Model",      18),
        ("MRR",          7),
        ("R@1",          7),
        ("R@5",          7),
        ("rho",          7),
        ("Trip.Acc",     9),
        ("Discr.",       7),
        ("Uniform.",     9),
    ]

    sep_line = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"
    header = "|" + "|".join(f" {name:^{w}} " for name, w in cols) + "|"

    print()
    print(sep_line)
    print(header)
    print(sep_line)

    for r in all_results:
        row = "|"
        row += f" {r['model']:<18} |"
        row += f" {r['mrr_avg']:>7.3f} |"
        row += f" {r['r1_avg']*100:>6.1f}% |"
        row += f" {(r['r5_en2sa']+r['r5_sa2en'])/2*100:>6.1f}% |"
        row += f" {r['spearman']:>7.3f} |"
        row += f" {r['triplet_acc']*100:>8.1f}% |"
        row += f" {r['triplet_margin']:>7.3f} |"
        row += f" {r['uniformity']:>9.3f} |"
        print(row)

    print(sep_line)

    # Detail: per-direction retrieval
    print("\nCross-lingual retrieval detail:")
    print(f"  {'Model':<18} {'En->Sa MRR':>10} {'R@1':>7} {'R@3':>7} {'R@5':>7}   {'Sa->En MRR':>10} {'R@1':>7} {'R@3':>7} {'R@5':>7}")
    print("  " + "-" * 95)
    for r in all_results:
        print(f"  {r['model']:<18} {r['mrr_en2sa']:>10.3f} {r['r1_en2sa']*100:>6.1f}% {r['r3_en2sa']*100:>6.1f}% {r['r5_en2sa']*100:>6.1f}%"
              f"   {r['mrr_sa2en']:>10.3f} {r['r1_sa2en']*100:>6.1f}% {r['r3_sa2en']*100:>6.1f}% {r['r5_sa2en']*100:>6.1f}%")

    # Detail: per-category triplet accuracy
    print("\nTriplet accuracy by category:")
    categories = ["mono_sa", "cross_lingual", "hard"]
    cat_header = f"  {'Model':<18}"
    for cat in categories:
        cat_header += f" {cat:>14}"
    print(cat_header)
    print("  " + "-" * (18 + 15 * len(categories)))
    for r in all_results:
        row = f"  {r['model']:<18}"
        for cat in categories:
            if cat in r["triplet_by_cat"]:
                c = r["triplet_by_cat"][cat]
                row += f" {c['correct']}/{c['total']} ({100*c['correct']/c['total']:>4.0f}%)"
            else:
                row += f" {'N/A':>14}"
        print(row)
    print()


def log_to_wandb(all_results):
    """Log comparison table and per-model metrics to W&B."""
    import wandb

    # Summary table
    columns = [
        "model", "mrr_avg", "r1_avg", "r5_avg",
        "spearman", "pearson",
        "triplet_acc", "triplet_margin",
        "mean_cos", "uniformity",
    ]
    table = wandb.Table(columns=columns)
    for r in all_results:
        r5_avg = (r["r5_en2sa"] + r["r5_sa2en"]) / 2
        table.add_data(
            r["model"], r["mrr_avg"], r["r1_avg"], r5_avg,
            r["spearman"], r["pearson"],
            r["triplet_acc"], r["triplet_margin"],
            r["mean_cos"], r["uniformity"],
        )
    wandb.log({"comparison_table": table})

    # Per-model scalar metrics
    for r in all_results:
        prefix = r["model"].replace(" ", "_").replace("(", "").replace(")", "")
        wandb.log({
            f"{prefix}/mrr_avg": r["mrr_avg"],
            f"{prefix}/r1_avg": r["r1_avg"],
            f"{prefix}/spearman": r["spearman"],
            f"{prefix}/triplet_acc": r["triplet_acc"],
            f"{prefix}/uniformity": r["uniformity"],
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark off-the-shelf embedding models against fine-tuned Sarvam-1 "
                    "on held-out Trika eval data"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to benchmark (default: all)",
    )
    parser.add_argument(
        "--model_path", type=str, default="../pretrained/sarvam-1",
        help="Path to Sarvam-1 base model",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to fine-tuned LoRA adapter (required if sarvam_ft in --models)",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="LM-STS-CFT",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_name", type=str, default="baseline-comparison",
        help="W&B run name",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for sentence-transformers (default: auto)",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: sarvam_ft requires adapter_path
    if "sarvam_ft" in args.models and args.adapter_path is None:
        logger.error("--adapter_path is required when benchmarking sarvam_ft")
        sys.exit(1)

    # Init W&B
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type="eval",
            config={
                "models": args.models,
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
            },
        )

    all_results = []

    for model_key in tqdm(args.models, desc="Models", unit="model"):
        info = MODEL_REGISTRY[model_key]
        logger.info(f"Loading {info['display']}...")
        start = time.time()

        if info["type"] == "sentence_transformer":
            device = args.device or "cuda"
            encoder = BaselineEncoder(
                info["hf_id"], device=device, prefix=info.get("prefix", ""),
            )
        else:  # causal_lm
            from model.causal_lm import CausalLMEncoder
            adapter = args.adapter_path if model_key == "sarvam_ft" else None
            encoder = CausalLMEncoder(
                model_path=args.model_path, adapter_path=adapter,
            )

        load_time = time.time() - start
        logger.info(f"  Loaded in {load_time:.1f}s")

        result = evaluate_model(encoder, info["display"])
        all_results.append(result)

        # Free memory
        del encoder
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()

    # Output
    print_comparison_table(all_results)

    if not args.no_wandb:
        log_to_wandb(all_results)
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

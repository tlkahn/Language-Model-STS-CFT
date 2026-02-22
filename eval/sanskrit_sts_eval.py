import argparse
import json
import logging
import sys
import time

import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add mteb model directory to path for CausalLMEncoder import
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / 'mteb'))
from model.causal_lm import CausalLMEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanskrit_sts_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sanskrit STS on VBT pairs")
    parser.add_argument("--model_path", type=str, default="../pretrained/sarvam-1",
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to JSON eval pairs file")
    parser.add_argument("--wandb_project", type=str, default="LM-STS-CFT",
                        help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default="sanskrit-sts-eval",
                        help="W&B run name")
    return parser.parse_args()


def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def main():
    args = parse_args()

    with open(args.eval_data) as f:
        pairs = json.load(f)

    logger.info(f"Loaded {len(pairs)} eval pairs")

    model = CausalLMEncoder(model_path=args.model_path, adapter_path=args.adapter_path)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="eval",
        config={
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "eval_data": args.eval_data,
        },
    )

    # Collect unique texts and encode once
    unique_texts = set()
    for p in pairs:
        unique_texts.add(p["text_a"])
        unique_texts.add(p["text_b"])
    unique_texts = sorted(unique_texts)

    logger.info(f"Encoding {len(unique_texts)} unique texts...")
    start = time.time()
    embeddings = model.encode(unique_texts)
    elapsed = time.time() - start
    logger.info(f"Encoded {len(unique_texts)} texts in {elapsed:.1f}s")

    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    # Compute cosine similarity per pair
    similarities = []
    labels = []
    for p in tqdm(pairs, desc="Computing similarities", unit="pair"):
        emb_a = embeddings[text_to_idx[p["text_a"]]]
        emb_b = embeddings[text_to_idx[p["text_b"]]]
        sim = cosine_similarity(emb_a, emb_b)
        similarities.append(sim)
        labels.append(p["label"])

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Compute metrics
    sim_mask = labels == 1
    dissim_mask = labels == 0

    mean_sim_similar = float(similarities[sim_mask].mean()) if sim_mask.any() else 0.0
    mean_sim_dissimilar = float(similarities[dissim_mask].mean()) if dissim_mask.any() else 0.0
    discrimination = mean_sim_similar - mean_sim_dissimilar
    auc_roc = float(roc_auc_score(labels, similarities))

    logger.info(f"Mean similarity (similar pairs):    {mean_sim_similar:.4f}")
    logger.info(f"Mean similarity (dissimilar pairs): {mean_sim_dissimilar:.4f}")
    logger.info(f"Discrimination (delta):             {discrimination:.4f}")
    logger.info(f"AUC-ROC:                            {auc_roc:.4f}")

    wandb.log({
        "mean_sim_similar": mean_sim_similar,
        "mean_sim_dissimilar": mean_sim_dissimilar,
        "discrimination": discrimination,
        "auc_roc": auc_roc,
    })

    table = wandb.Table(columns=["metric", "value"])
    table.add_data("mean_sim_similar", mean_sim_similar)
    table.add_data("mean_sim_dissimilar", mean_sim_dissimilar)
    table.add_data("discrimination", discrimination)
    table.add_data("auc_roc", auc_roc)
    wandb.log({"sanskrit_sts_summary": table})

    wandb.finish()


if __name__ == "__main__":
    main()

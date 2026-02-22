"""One-off script to extract per-triplet margins, per-pair STS cosines, and
pairwise cosine stats for all models. Prints TSV tables to stdout."""

import sys
import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "mteb")
warnings.filterwarnings("ignore")

from trika_eval_data import get_all_verses, get_sts_pairs, get_triplets


def encode(model, prefix, texts):
    if hasattr(model, "_encode_single"):  # CausalLMEncoder
        return np.array(model.encode(texts))
    return model.encode([prefix + t for t in texts], show_progress_bar=False)


def main():
    from sentence_transformers import SentenceTransformer
    from model.causal_lm import CausalLMEncoder

    models = {}
    for name, hf_id, prefix in [
        ("LaBSE", "sentence-transformers/LaBSE", ""),
        ("E5", "intfloat/multilingual-e5-large", "query: "),
        ("BGE-M3", "BAAI/bge-m3", ""),
        ("Vyakyarth", "krutrim-ai-labs/Vyakyarth", ""),
    ]:
        m = SentenceTransformer(hf_id, device="cuda")
        models[name] = (m, prefix)

    models["Base"] = (
        CausalLMEncoder(model_path="../pretrained/sarvam-1"),
        "",
    )
    models["FT"] = (
        CausalLMEncoder(
            model_path="../pretrained/sarvam-1",
            adapter_path="../train/output/20260222141912",
        ),
        "",
    )

    names = list(models.keys())

    # --- Per-triplet margins ---
    triplets = get_triplets()
    print("=== PER-TRIPLET MARGINS (pos_sim - neg_sim) ===")
    header = f"{'Category':<15} {'Triplet':<45}"
    for n in names:
        header += f" {n:>10}"
    print(header)
    print("-" * len(header))

    cat_margins = {n: {} for n in names}
    for label, anchor, positive, negative, category in triplets:
        row = f"{category:<15} {label:<45}"
        for n in names:
            model, prefix = models[n]
            embs = encode(model, prefix, [anchor, positive, negative])
            pos_sim = cosine_similarity([embs[0]], [embs[1]])[0, 0]
            neg_sim = cosine_similarity([embs[0]], [embs[2]])[0, 0]
            margin = pos_sim - neg_sim
            row += f" {margin:>+10.4f}"
            cat_margins[n].setdefault(category, []).append(margin)
        print(row)

    # Per-category margin summary
    print()
    print("=== MARGIN SUMMARY BY CATEGORY ===")
    categories = ["mono_sa", "cross_lingual", "hard"]
    header2 = f"{'Model':<12}"
    for cat in categories:
        header2 += f" {cat + ' mean':>16} {cat + ' min':>16}"
    header2 += f" {'overall mean':>16}"
    print(header2)
    print("-" * len(header2))
    for n in names:
        row = f"{n:<12}"
        all_m = []
        for cat in categories:
            margins = cat_margins[n].get(cat, [])
            all_m.extend(margins)
            row += f" {np.mean(margins):>+16.4f} {np.min(margins):>+16.4f}"
        row += f" {np.mean(all_m):>+16.4f}"
        print(row)

    # --- STS per-pair cosines ---
    sts_pairs = get_sts_pairs()
    print()
    print("=== STS PER-PAIR COSINES ===")
    header3 = f"{'Score':>5} {'Label':<50}"
    for n in names:
        header3 += f" {n:>10}"
    print(header3)
    print("-" * len(header3))

    for text_a, text_b, score, label in sorted(
        sts_pairs, key=lambda x: x[2], reverse=True
    ):
        row = f"{score:>5.1f} {label:<50}"
        for n in names:
            model, prefix = models[n]
            embs = encode(model, prefix, [text_a, text_b])
            sim = cosine_similarity([embs[0]], [embs[1]])[0, 0]
            row += f" {sim:>10.4f}"
        print(row)

    # --- Pairwise cosine stats ---
    sa_ids, sa_texts, _, _ = get_all_verses()
    print()
    print("=== PAIRWISE COSINE STATS (32 Sa verses) ===")
    print(f"{'Model':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-" * 56)
    for n in names:
        model, prefix = models[n]
        embs = encode(model, prefix, sa_texts)
        sim_mat = cosine_similarity(embs)
        triu = sim_mat[np.triu_indices(len(embs), k=1)]
        print(
            f"{n:<12} {np.mean(triu):>8.4f} {np.std(triu):>8.4f} "
            f"{np.min(triu):>8.4f} {np.max(triu):>8.4f} "
            f"{np.max(triu) - np.min(triu):>8.4f}"
        )


if __name__ == "__main__":
    main()

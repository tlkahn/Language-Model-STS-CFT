"""Generate contrastive triplets from the VBT corpus for Stage 2 fine-tuning.

Uses four strategies to create ~600-640 triplets from 168 VBT verses:
  A — Direct from similarity pairs (with multiple Itihasa negatives)
  B — Combinatorial within-domain expansion
  C — Cross-lingual Sa->En (VBT verse -> VBT translation, Itihasa En negative)
  D — Reverse cross-lingual En->Sa (VBT translation -> VBT verse, Itihasa Sa negative)

All hard negatives are sourced from the Itihasa corpus (rahular/itihasa) rather than
cross-domain VBT verses, because all 168 VBT verses discuss dharana practices and
share significant thematic overlap even across domains.

Domain assignment uses union-find over VBT_SIMILARITY_PAIRS to identify practice
domains (breath, kundalini, void, sound, gaze, bliss, mind, non-dual, worship).

Usage:
    python vbt_triplets.py                              # all strategies, ~630 triplets
    python vbt_triplets.py --strategies A C             # subset of strategies
    python vbt_triplets.py --n_negs 5                   # more negatives per sim pair
"""

import argparse
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def load_vbt_corpus(sa_embedding_path):
    """Import VBT data from the sa-embedding project."""
    sys.path.insert(0, sa_embedding_path)
    from vbt_corpus import (
        VBT_CORPUS, VBT_TRANSLATIONS,
        VBT_SIMILARITY_PAIRS, VBT_DISSIMILARITY_PAIRS,
    )
    return VBT_CORPUS, VBT_TRANSLATIONS, VBT_SIMILARITY_PAIRS, VBT_DISSIMILARITY_PAIRS


def load_itihasa_negatives():
    """Load rahular/itihasa, return (all_sanskrit, all_english) lists for negative sampling."""
    print("Loading Itihasa corpus for hard negatives...")
    ds = load_dataset("rahular/itihasa")
    all_splits = concatenate_datasets([ds[split] for split in ds.keys()])

    sn_list = [row["sn"] for row in all_splits["translation"]]
    en_list = [row["en"] for row in all_splits["translation"]]
    print(f"  Loaded {len(sn_list)} Itihasa entries for negative sampling")
    return sn_list, en_list


def build_domain_map(sim_pairs, corpus):
    """Union-find over verses appearing in similarity pairs -> {verse_idx: domain_id}.

    Two verses sharing a similarity pair belong to the same domain.
    Connected components = practice domains.
    """
    # Map verse text to corpus index
    text_to_idx = {text: i for i, text in enumerate(corpus)}

    # Build adjacency from similarity pairs (only Sa-Sa pairs)
    parent = list(range(len(corpus)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for text_a, text_b in sim_pairs:
        idx_a = text_to_idx.get(text_a)
        idx_b = text_to_idx.get(text_b)
        if idx_a is not None and idx_b is not None:
            union(idx_a, idx_b)

    # Assign domain IDs: verses in sim pairs get their root as domain, others get -1
    verses_in_pairs = set()
    for text_a, text_b in sim_pairs:
        idx_a = text_to_idx.get(text_a)
        idx_b = text_to_idx.get(text_b)
        if idx_a is not None:
            verses_in_pairs.add(idx_a)
        if idx_b is not None:
            verses_in_pairs.add(idx_b)

    domain_map = {}
    for idx in range(len(corpus)):
        if idx in verses_in_pairs:
            domain_map[idx] = find(idx)
        else:
            domain_map[idx] = -1

    return domain_map


def assign_corpus_domains(corpus, domain_map):
    """Return {domain_id: [corpus_indices]} for domains with >= 2 verses."""
    domains = {}
    for idx in range(len(corpus)):
        d = domain_map[idx]
        if d == -1:
            continue
        domains.setdefault(d, []).append(idx)

    # Filter out singletons
    return {d: indices for d, indices in domains.items() if len(indices) >= 2}


def strategy_a(sim_pairs, corpus, translations, itihasa_sn, itihasa_en, rng, n_negs):
    """Direct from similarity pairs: generate n_negs triplets per pair with Itihasa negatives."""
    text_to_idx = {text: i for i, text in enumerate(corpus)}
    trans_set = set(translations)
    triplets = []

    start_time = time.time()
    for text_a, text_b in tqdm(sim_pairs, desc="Strategy A (sim pairs)", unit="pair"):
        for _ in range(n_negs):
            # Determine if this is a cross-lingual pair
            a_is_sa = text_a in text_to_idx
            b_is_en = text_b in trans_set

            if a_is_sa and b_is_en:
                # Cross-lingual: Sa anchor, En positive -> En Itihasa negative
                neg = itihasa_en[rng.integers(0, len(itihasa_en))]
            else:
                # Sa-Sa pair -> Sa Itihasa negative
                neg = itihasa_sn[rng.integers(0, len(itihasa_sn))]

            triplets.append({"sent0": text_a, "sent1": text_b, "hard_neg": neg})

    elapsed = time.time() - start_time
    print(f"  Strategy A: {len(triplets)} triplets in {elapsed:.1f}s")
    return triplets


def strategy_b(corpus, corpus_domains, existing_pairs_set, itihasa_sn, rng):
    """Combinatorial within-domain expansion: all pairs within each domain, dedup against existing."""
    triplets = []

    start_time = time.time()
    domain_items = sorted(corpus_domains.items())
    for _, indices in tqdm(domain_items, desc="Strategy B (within-domain)", unit="domain"):
        for i, j in combinations(indices, 2):
            pair_key = (min(corpus[i], corpus[j]), max(corpus[i], corpus[j]))
            if pair_key in existing_pairs_set:
                continue
            neg = itihasa_sn[rng.integers(0, len(itihasa_sn))]
            triplets.append({"sent0": corpus[i], "sent1": corpus[j], "hard_neg": neg})

    elapsed = time.time() - start_time
    print(f"  Strategy B: {len(triplets)} triplets in {elapsed:.1f}s")
    return triplets


def strategy_c(corpus, translations, itihasa_en, rng):
    """Cross-lingual Sa->En: (VBT_CORPUS[i], VBT_TRANSLATIONS[i], itihasa_en[j])."""
    triplets = []

    start_time = time.time()
    for i in tqdm(range(len(corpus)), desc="Strategy C (Sa->En)", unit="verse"):
        neg = itihasa_en[rng.integers(0, len(itihasa_en))]
        triplets.append({
            "sent0": corpus[i],
            "sent1": translations[i],
            "hard_neg": neg,
        })

    elapsed = time.time() - start_time
    print(f"  Strategy C: {len(triplets)} triplets in {elapsed:.1f}s")
    return triplets


def strategy_d(corpus, translations, itihasa_sn, rng):
    """Reverse cross-lingual En->Sa: (VBT_TRANSLATIONS[i], VBT_CORPUS[i], itihasa_sn[j])."""
    triplets = []

    start_time = time.time()
    for i in tqdm(range(len(corpus)), desc="Strategy D (En->Sa)", unit="verse"):
        neg = itihasa_sn[rng.integers(0, len(itihasa_sn))]
        triplets.append({
            "sent0": translations[i],
            "sent1": corpus[i],
            "hard_neg": neg,
        })

    elapsed = time.time() - start_time
    print(f"  Strategy D: {len(triplets)} triplets in {elapsed:.1f}s")
    return triplets


def deduplicate(triplets):
    """Remove triplets with identical (sent0, sent1) pairs, keeping first occurrence."""
    seen = set()
    unique = []
    for t in triplets:
        key = (t["sent0"], t["sent1"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Generate VBT contrastive triplets for Stage 2 fine-tuning")
    parser.add_argument("--output_csv", type=str, default="vbt_triplets.csv",
                        help="Output CSV path (default: vbt_triplets.csv)")
    parser.add_argument("--sa_embedding_path", type=str,
                        default="/Users/toeinriver/Projects/sa-embedding",
                        help="Path to sa-embedding project")
    parser.add_argument("--strategies", type=str, nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"],
                        help="Which strategies to run (default: A B C D)")
    parser.add_argument("--n_negs", type=int, default=3,
                        help="Negatives per similarity pair for strategy A (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load VBT data
    corpus, translations, sim_pairs, dissim_pairs = load_vbt_corpus(args.sa_embedding_path)
    print(f"VBT corpus: {len(corpus)} verses, {len(sim_pairs)} similarity pairs, "
          f"{len(dissim_pairs)} dissimilarity pairs")

    # Load Itihasa for hard negatives
    itihasa_sn, itihasa_en = load_itihasa_negatives()

    # Build domain map
    domain_map = build_domain_map(sim_pairs, corpus)
    corpus_domains = assign_corpus_domains(corpus, domain_map)
    n_domains = len(corpus_domains)
    domain_sizes = [len(v) for v in corpus_domains.values()]
    n_uncategorized = sum(1 for v in domain_map.values() if v == -1)
    print(f"Domain map: {n_domains} domains (sizes: {sorted(domain_sizes, reverse=True)}), "
          f"{n_uncategorized} uncategorized verses")

    # Build existing pairs set for dedup in strategy B
    existing_pairs_set = set()
    for text_a, text_b in sim_pairs:
        existing_pairs_set.add((min(text_a, text_b), max(text_a, text_b)))

    # Run strategies
    all_triplets = []

    if "A" in args.strategies:
        all_triplets.extend(
            strategy_a(sim_pairs, corpus, translations,
                       itihasa_sn, itihasa_en, rng, args.n_negs))

    if "B" in args.strategies:
        all_triplets.extend(
            strategy_b(corpus, corpus_domains, existing_pairs_set, itihasa_sn, rng))

    if "C" in args.strategies:
        all_triplets.extend(
            strategy_c(corpus, translations, itihasa_en, rng))

    if "D" in args.strategies:
        all_triplets.extend(
            strategy_d(corpus, translations, itihasa_sn, rng))

    print(f"\nTotal before dedup: {len(all_triplets)}")
    all_triplets = deduplicate(all_triplets)
    print(f"Total after dedup: {len(all_triplets)}")

    # Shuffle and save
    df = pd.DataFrame(all_triplets)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Drop any rows with empty values
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    df = df[df["sent0"].str.strip().astype(bool)
            & df["sent1"].str.strip().astype(bool)
            & df["hard_neg"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with empty values")

    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} triplets to {args.output_csv}")


if __name__ == "__main__":
    main()

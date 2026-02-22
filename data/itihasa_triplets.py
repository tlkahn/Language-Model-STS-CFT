"""Generate contrastive triplets from the Itihasa parallel corpus.

Downloads rahular/itihasa from HuggingFace (Sanskrit-English parallel pairs from
Ramayana + Mahabharata) and generates two types of triplets:

- Cross-lingual: (Sanskrit, English translation, distant English translation)
- Monolingual Sanskrit: (Sanskrit[i], Sanskrit[i+1], distant Sanskrit)

Hard negatives are sampled from positions >= min_distance away to ensure
thematic distance without needing chapter metadata.

Usage:
    python itihasa_triplets.py                          # ~167K triplets
    python itihasa_triplets.py --num_rows 1000          # pilot mode
    python itihasa_triplets.py --triplet_types mono_sa  # Sanskrit only
"""

import argparse
import time

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def load_itihasa(num_rows=None):
    """Load rahular/itihasa, concat all splits, flatten to sn/en columns with position."""
    ds = load_dataset("rahular/itihasa")
    all_splits = concatenate_datasets([ds[split] for split in ds.keys()])

    if num_rows is not None:
        all_splits = all_splits.select(range(min(num_rows, len(all_splits))))

    df = pd.DataFrame({
        "sn": [row["sn"] for row in all_splits["translation"]],
        "en": [row["en"] for row in all_splits["translation"]],
    })
    df["position"] = range(len(df))
    return df


def sample_distant_negative(position, total, min_distance, rng):
    """Sample an index >= min_distance away from position."""
    low_end = position - min_distance
    high_start = position + min_distance

    valid_ranges = []
    if low_end >= 0:
        valid_ranges.append((0, low_end))
    if high_start < total:
        valid_ranges.append((high_start, total - 1))

    if not valid_ranges:
        # Fallback: dataset too small relative to min_distance, sample any other index
        idx = rng.integers(0, total - 1)
        return idx if idx < position else idx + 1

    # Pick a range weighted by size, then sample within it
    sizes = [hi - lo + 1 for lo, hi in valid_ranges]
    total_size = sum(sizes)
    r = rng.integers(0, total_size)
    cumulative = 0
    for (lo, hi), size in zip(valid_ranges, sizes):
        cumulative += size
        if r < cumulative:
            return rng.integers(lo, hi + 1)

    return valid_ranges[-1][0]


def generate_cross_lingual(df, min_distance, rng):
    """Generate cross-lingual triplets: (sn[i], en[i], en[j]) where |i-j| >= min_distance."""
    total = len(df)
    triplets = []

    start_time = time.time()
    for i in tqdm(range(total), desc="Cross-lingual triplets", unit="pair"):
        j = sample_distant_negative(i, total, min_distance, rng)
        triplets.append({
            "sent0": df.iloc[i]["sn"],
            "sent1": df.iloc[i]["en"],
            "hard_neg": df.iloc[j]["en"],
        })

    elapsed = time.time() - start_time
    print(f"  Generated {len(triplets)} cross-lingual triplets in {elapsed:.1f}s "
          f"({len(triplets)/elapsed:.0f} triplets/s)")
    return pd.DataFrame(triplets)


def generate_monolingual_sa(df, min_distance, rng):
    """Generate monolingual Sanskrit triplets: (sn[i], sn[i+1], sn[j]) where |i-j| >= min_distance."""
    total = len(df)
    triplets = []

    start_time = time.time()
    for i in tqdm(range(total - 1), desc="Monolingual Sa triplets", unit="pair"):
        j = sample_distant_negative(i, total, min_distance, rng)
        triplets.append({
            "sent0": df.iloc[i]["sn"],
            "sent1": df.iloc[i + 1]["sn"],
            "hard_neg": df.iloc[j]["sn"],
        })

    elapsed = time.time() - start_time
    print(f"  Generated {len(triplets)} monolingual Sanskrit triplets in {elapsed:.1f}s "
          f"({len(triplets)/elapsed:.0f} triplets/s)")
    return pd.DataFrame(triplets)


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive triplets from the Itihasa parallel corpus")
    parser.add_argument("--output_csv", type=str, default="itihasa_triplets.csv",
                        help="Output CSV path (default: itihasa_triplets.csv)")
    parser.add_argument("--triplet_types", type=str, default="both",
                        choices=["cross_lingual", "mono_sa", "both"],
                        help="Triplet types to generate (default: both)")
    parser.add_argument("--min_distance", type=int, default=100,
                        help="Min positional distance for hard negatives (default: 100)")
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Limit input rows (pilot mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading Itihasa dataset...")
    df = load_itihasa(args.num_rows)
    print(f"  Loaded {len(df)} parallel pairs")

    frames = []

    if args.triplet_types in ("cross_lingual", "both"):
        frames.append(generate_cross_lingual(df, args.min_distance, rng))

    if args.triplet_types in ("mono_sa", "both"):
        frames.append(generate_monolingual_sa(df, args.min_distance, rng))

    result = pd.concat(frames, ignore_index=True)

    # Shuffle
    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Drop any rows with empty values
    before = len(result)
    result = result.dropna().reset_index(drop=True)
    result = result[result["sent0"].str.strip().astype(bool)
                    & result["sent1"].str.strip().astype(bool)
                    & result["hard_neg"].str.strip().astype(bool)].reset_index(drop=True)
    if len(result) < before:
        print(f"  Dropped {before - len(result)} rows with empty values")

    result.to_csv(args.output_csv, index=False)
    print(f"Saved {len(result)} triplets to {args.output_csv}")


if __name__ == "__main__":
    main()

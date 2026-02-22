"""Convert VBT corpus similarity/dissimilarity pairs to JSON eval format.

Reads VBT_SIMILARITY_PAIRS and VBT_DISSIMILARITY_PAIRS from the sa-embedding
project and writes eval/vbt_eval_pairs.json.

Usage:
    python vbt_to_json.py
"""

import json
import sys

sys.path.insert(0, "/Users/toeinriver/Projects/sa-embedding")
from vbt_corpus import VBT_SIMILARITY_PAIRS, VBT_DISSIMILARITY_PAIRS

pairs = []

for text_a, text_b in VBT_SIMILARITY_PAIRS:
    pairs.append({"text_a": text_a, "text_b": text_b, "label": 1})

for text_a, text_b in VBT_DISSIMILARITY_PAIRS:
    pairs.append({"text_a": text_a, "text_b": text_b, "label": 0})

out_path = "vbt_eval_pairs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(pairs)} pairs ({sum(1 for p in pairs if p['label'] == 1)} similar, "
      f"{sum(1 for p in pairs if p['label'] == 0)} dissimilar) to {out_path}")

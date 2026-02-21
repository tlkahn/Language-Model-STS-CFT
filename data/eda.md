---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# EDA: Preprocessed NLI Dataset

Quick look at the tokenized NLI triplets in `data/processed/`.

```python
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np

ds = load_from_disk("processed/")
ds.reset_format()  # expose text columns hidden by set_format
print(f"Rows: {len(ds):,}")
print(f"Columns: {ds.column_names}")
```

## Sample triplets

```python
import pandas as pd

df = ds.select(range(5)).to_pandas()
df[["sent0", "sent1", "hard_neg"]]
```

## Text length distributions (in characters)

```python
# sample 20k rows for speed
sample = ds.shuffle(seed=42).select(range(20_000))

char_lens = {
    col: [len(s) for s in sample[col]]
    for col in ["sent0", "sent1", "hard_neg"]
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (col, lens) in zip(axes, char_lens.items()):
    ax.hist(lens, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel("char length")
    ax.axvline(np.median(lens), color="red", linestyle="--", label=f"median={np.median(lens):.0f}")
    ax.legend()
axes[0].set_ylabel("count")
plt.suptitle("Character length distributions (20k sample)")
plt.tight_layout()
plt.show()
```

## Token length distributions (non-padding tokens)

```python
token_lens = {
    col: [int(sum(m)) for m in sample[f"{col}_attention_mask"]]
    for col in ["sent0", "sent1", "hard_neg"]
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (col, lens) in zip(axes, token_lens.items()):
    ax.hist(lens, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel("token count")
    ax.axvline(np.median(lens), color="red", linestyle="--", label=f"median={np.median(lens):.0f}")
    ax.legend()
axes[0].set_ylabel("count")
plt.suptitle("Token count distributions (20k sample, max_length=150)")
plt.tight_layout()
plt.show()
```

## Truncation check

How many sequences hit the `max_length=150` ceiling?

```python
MAX_LEN = 150
for col in ["sent0", "sent1", "hard_neg"]:
    n_max = sum(1 for t in token_lens[col] if t >= MAX_LEN)
    pct = n_max / len(token_lens[col]) * 100
    print(f"{col}: {n_max}/{len(token_lens[col])} ({pct:.2f}%) hit max_length={MAX_LEN}")
```

## Summary statistics

```python
rows = []
for col in ["sent0", "sent1", "hard_neg"]:
    cl = np.array(char_lens[col])
    tl = np.array(token_lens[col])
    rows.append({
        "column": col,
        "char_mean": f"{cl.mean():.1f}",
        "char_median": f"{np.median(cl):.0f}",
        "char_max": cl.max(),
        "tok_mean": f"{tl.mean():.1f}",
        "tok_median": f"{np.median(tl):.0f}",
        "tok_max": tl.max(),
        "tok_p95": f"{np.percentile(tl, 95):.0f}",
    })
pd.DataFrame(rows).set_index("column")
```

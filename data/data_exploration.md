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

# NLI Data Preprocessing Exploration

Walk through the data preprocessing pipeline step by step:
1. Raw CSV triplets (anchor, positive, hard negative)
2. Tokenization with prefix-renaming
3. Final tensor format consumed by `ContrastiveTrainer`

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 1. Raw CSV Data
<!-- #endregion -->

```python
import pandas as pd
from datasets import load_dataset

# Quick peek with pandas
df = pd.read_csv('nli_for_simcse.csv')
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
df.head(3)
```

```python
# Look at a single triplet in full
row = df.iloc[1]
print("ANCHOR (sent0):")
print(f"  {row['sent0']}")
print()
print("POSITIVE (sent1) — entailment/paraphrase:")
print(f"  {row['sent1']}")
print()
print("HARD NEGATIVE (hard_neg) — contradiction:")
print(f"  {row['hard_neg']}")
```

```python
# Sentence length distributions (in words)
for col in ['sent0', 'sent1', 'hard_neg']:
    lengths = df[col].str.split().str.len()
    print(f"{col:10s}  mean={lengths.mean():.1f}  median={lengths.median():.0f}  "
          f"max={lengths.max()}  min={lengths.min()}")
```

## 2. Tokenization Step-by-Step

The preprocessor tokenizes each column independently, then renames keys with a prefix.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../pretrained/MiniCPM-2B-dpo-bf16/', local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocab size: {tokenizer.vocab_size:,}")
print(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"Pad token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
print(f"add_eos_token: {tokenizer.add_eos_token}")
```

```python
# Tokenize a single sentence to see what happens
sample_text = row['sent0']
print(f"Text: {sample_text!r}")
print()

# Without padding (to see real token count)
tokens_raw = tokenizer(sample_text)
print(f"Token count (no padding): {len(tokens_raw['input_ids'])}")
print(f"Token IDs: {tokens_raw['input_ids']}")
print()

# Decode back to see individual tokens
token_strings = [tokenizer.decode(tid) for tid in tokens_raw['input_ids']]
print(f"Decoded tokens: {token_strings}")
print(f"Last token: {token_strings[-1]!r} (should be EOS)")
```

```python
# With max_length=150 padding (what the preprocessor does)
tokens_padded = tokenizer(sample_text, padding='max_length', truncation=True,
                          return_tensors='pt', max_length=150)

input_ids = tokens_padded['input_ids'][0]
attention_mask = tokens_padded['attention_mask'][0]

real_tokens = attention_mask.sum().item()
pad_tokens = (attention_mask == 0).sum().item()

print(f"Shape: {input_ids.shape}")
print(f"Real tokens: {real_tokens}, Padding tokens: {pad_tokens}")
print()
print(f"First 20 IDs:    {input_ids[:20].tolist()}")
print(f"Attention mask:  {attention_mask[:20].tolist()}")
print()
print(f"Last 10 IDs:     {input_ids[-10:].tolist()}")
print(f"Attention mask:  {attention_mask[-10:].tolist()}")
print(f"\nPad token ID = {tokenizer.pad_token_id} (same as EOS = {tokenizer.eos_token_id})")
```

```python
# The _tokenize method renames keys with a prefix
# This is how 3 sentences coexist in the same dataset row

def tokenize_with_prefix(text, prefix):
    """Mirrors NLIPreprocess._tokenize()"""
    out = tokenizer(text, padding='max_length', truncation=True,
                    return_tensors='pt', max_length=150)
    out[f'{prefix}_input_ids'] = out.pop('input_ids')
    out[f'{prefix}_attention_mask'] = out.pop('attention_mask')
    return out

result = tokenize_with_prefix(sample_text, 'sent0')
print(f"Keys after renaming: {list(result.keys())}")
print(f"sent0_input_ids shape: {result['sent0_input_ids'].shape}")
```

## 3. Token Length Distribution

Check how many sentences get truncated at `max_length=150`.

```python
from tqdm import tqdm

# Sample 5000 rows for speed
sample_df = df.sample(n=min(5000, len(df)), random_state=42)

token_lengths = {col: [] for col in ['sent0', 'sent1', 'hard_neg']}

for _, r in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Tokenizing sample", unit="row"):
    for col in token_lengths:
        toks = tokenizer(r[col], add_special_tokens=True)
        token_lengths[col].append(len(toks['input_ids']))

for col, lengths in token_lengths.items():
    lengths_arr = pd.Series(lengths)
    truncated = (lengths_arr >= 150).sum()
    print(f"{col:10s}  mean={lengths_arr.mean():.1f}  p95={lengths_arr.quantile(0.95):.0f}  "
          f"max={lengths_arr.max()}  truncated@150={truncated} ({100*truncated/len(lengths_arr):.1f}%)")
```

## 4. Processed Dataset (what the trainer sees)

Load the preprocessed Arrow dataset from `data/processed/`.

```python
from datasets import load_from_disk

ds = load_from_disk('./processed/')
print(f"Dataset: {ds}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")
```

```python
# Look at a single processed example
sample = ds[0]
for key in sorted(sample.keys()):
    val = sample[key]
    if hasattr(val, 'shape'):
        print(f"{key:30s}  shape={val.shape}  dtype={val.dtype}")
    else:
        print(f"{key:30s}  type={type(val).__name__}  len={len(val) if hasattr(val, '__len__') else 'N/A'}")
```

```python
# Decode the processed tokens back to text to verify correctness
sample = ds[1]

for prefix in ['sent0', 'sent1', 'hard_neg']:
    ids = sample[f'{prefix}_input_ids']
    mask = sample[f'{prefix}_attention_mask']
    real_len = mask.sum().item() if hasattr(mask, 'sum') else sum(mask)
    
    # Decode only real tokens (not padding)
    real_ids = ids[:real_len] if hasattr(ids, '__getitem__') else ids
    decoded = tokenizer.decode(real_ids, skip_special_tokens=True)
    
    print(f"{prefix} ({real_len} tokens):")
    print(f"  {decoded}")
    print()
```

<!-- #region -->
## 5. How the Trainer Consumes This

The `ContrastiveTrainer.compute_loss()` unpacks each batch into 3 dicts:

```python
sent0 = {'input_ids': inputs['sent0_input_ids'],
         'attention_mask': inputs['sent0_attention_mask']}
# ... same for sent1, hard_neg

sent0_embed = model(**sent0, output_hidden_states=True).hidden_states[-1][:, -1, :]
```

Each gets fed through the model separately. The **last hidden state at the last token position** (EOS) is used as the embedding. Then InfoNCE loss is computed over (anchor, positive, hard_negative) triplets.
<!-- #endregion -->

```python
import torch

# Simulate what a DataLoader batch looks like
batch_size = 4
batch = ds[:batch_size]

# The trainer unpacks it like this:
sent0 = {'input_ids': torch.tensor(batch['sent0_input_ids']),
         'attention_mask': torch.tensor(batch['sent0_attention_mask'])}
sent1 = {'input_ids': torch.tensor(batch['sent1_input_ids']),
         'attention_mask': torch.tensor(batch['sent1_attention_mask'])}
hard_neg = {'input_ids': torch.tensor(batch['hard_neg_input_ids']),
            'attention_mask': torch.tensor(batch['hard_neg_attention_mask'])}

print(f"Batch size: {batch_size}")
for name, d in [('sent0', sent0), ('sent1', sent1), ('hard_neg', hard_neg)]:
    print(f"{name:10s}  input_ids={d['input_ids'].shape}  attention_mask={d['attention_mask'].shape}")
```

```python
# Where is the EOS token (= the embedding extraction point) in each sequence?
# Since we pad to max_length and add_eos_token=True, EOS is the last real token.

for name, d in [('sent0', sent0), ('sent1', sent1), ('hard_neg', hard_neg)]:
    # Last real token position = sum of attention_mask - 1
    real_lengths = d['attention_mask'].sum(dim=1)
    last_positions = real_lengths - 1
    
    # Check that the token at that position is EOS
    for i in range(batch_size):
        pos = last_positions[i].item()
        token_id = d['input_ids'][i, pos].item()
        is_eos = token_id == tokenizer.eos_token_id
        print(f"{name}[{i}]: real_len={real_lengths[i].item()}, "
              f"last_token_pos={pos}, token_id={token_id}, is_EOS={is_eos}")
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Key Takeaways

- Each row has 3 sentences: **anchor** (sent0), **positive** (sent1), **hard negative** (hard_neg)
- All tokenized to fixed length 150 with padding, producing 6 columns: `{prefix}_input_ids` + `{prefix}_attention_mask`
- `set_format("torch")` makes the dataset return PyTorch tensors — no custom collator needed
- The model extracts embeddings from the **last token (EOS)** of the last hidden layer
- `pad_token = eos_token` means padding tokens have the same ID as EOS, but the **attention mask** distinguishes them — the model only attends to real tokens
<!-- #endregion -->

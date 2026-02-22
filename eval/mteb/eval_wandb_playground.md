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

# Eval W&B Playground

Explore MTEB result JSON structure and find the right wandb logging pattern.

**Problem:** The STS eval script logs per-task metrics to wandb, but the BIOSSES pane on the dashboard shows no data.

**Goal:** Figure out the correct way to parse intermediary results and log them so wandb renders charts properly.


## 1. Load and explore MTEB result JSON

```python
import json
from pathlib import Path

results_dir = Path("results/minicpm/sts")
result_files = sorted(results_dir.glob("*.json"))
print(f"Found {len(result_files)} result files:")
for f in result_files:
    print(f"  {f.name}")
```

```python
# Load first available result and show full structure
sample_file = result_files[0]
with open(sample_file) as f:
    sample = json.load(f)

print(f"=== {sample_file.name} ===")
print(json.dumps(sample, indent=2))
```

```python
# Show top-level keys and split keys for every result file
for f in result_files:
    with open(f) as fh:
        data = json.load(fh)
    top_keys = list(data.keys())
    # split keys are anything that's not metadata
    meta_keys = {"mteb_version", "dataset_revision", "mteb_dataset_name"}
    split_keys = [k for k in top_keys if k not in meta_keys]
    print(f"{f.stem:25s}  splits={split_keys}  metric_keys={list(data[split_keys[0]].keys()) if split_keys else 'N/A'}")
```

## 2. Simulate what `evaluation.run()` returns

The MTEB `.run()` method returns a dict keyed by task name, with the same structure as the JSON files.

```python
# Reconstruct what evaluation.run() returns from saved JSONs
all_results = {}
for f in result_files:
    with open(f) as fh:
        data = json.load(fh)
    task_name = data["mteb_dataset_name"]
    all_results[task_name] = data

print(f"Tasks loaded: {list(all_results.keys())}")
print()

# Show how our current parsing works
for task_name, task_data in all_results.items():
    # This is what the eval script does:
    task_scores = task_data  # results.get(task) returns this
    
    # Try splits in order
    metrics = None
    for split in ("test", "validation"):
        if split in task_scores:
            metrics = task_scores[split]
            break
    
    if metrics:
        cos_sim = metrics.get("cos_sim", {})
        spearman = cos_sim.get("spearman")
        pearson = cos_sim.get("pearson")
        print(f"{task_name:20s}  spearman={spearman:.4f}  pearson={pearson:.4f}")
```

<!-- #region -->
## 3. W&B logging experiments

The current script does:
```python
wandb.log({f"{task}/cos_sim_spearman": spearman})
```

This creates separate wandb steps per task. Possible issues:
- Each `wandb.log()` call increments the step counter, so metrics end up at different x-axis positions
- wandb auto-creates panels per unique metric prefix, but single-point metrics may not render well

Let's try different approaches.
<!-- #endregion -->

```python
import wandb
```

### Approach A: Current approach — per-task `wandb.log()` calls (one step per task)

Each task gets its own step. Only one metric per step has a value; the rest are missing. wandb may not render this well.

```python
run_a = wandb.init(project="LM-STS-CFT", name="logging-test-A-per-step", job_type="debug", reinit=True)

for task_name, task_data in all_results.items():
    metrics = task_data.get("test", task_data.get("validation", {}))
    cos_sim = metrics.get("cos_sim", {})
    spearman = cos_sim.get("spearman")
    pearson = cos_sim.get("pearson")
    if spearman is not None:
        wandb.log({f"{task_name}/cos_sim_spearman": spearman, f"{task_name}/cos_sim_pearson": pearson})

wandb.finish()
print(f"Run A URL: {run_a.url}")
```

### Approach B: Single `wandb.log()` with all tasks at once

Log everything in one step so all metrics coexist.

```python
run_b = wandb.init(project="LM-STS-CFT", name="logging-test-B-single-step", job_type="debug", reinit=True)

log_dict = {}
for task_name, task_data in all_results.items():
    metrics = task_data.get("test", task_data.get("validation", {}))
    cos_sim = metrics.get("cos_sim", {})
    spearman = cos_sim.get("spearman")
    pearson = cos_sim.get("pearson")
    if spearman is not None:
        log_dict[f"{task_name}/cos_sim_spearman"] = spearman
        log_dict[f"{task_name}/cos_sim_pearson"] = pearson

wandb.log(log_dict)
wandb.finish()
print(f"Run B URL: {run_b.url}")
```

### Approach C: `wandb.summary` (no step history, just final values)

Summary metrics appear in the run overview and are ideal for single-value metrics like eval scores.

```python
run_c = wandb.init(project="LM-STS-CFT", name="logging-test-C-summary", job_type="debug", reinit=True)

all_spearman = {}
for task_name, task_data in all_results.items():
    metrics = task_data.get("test", task_data.get("validation", {}))
    cos_sim = metrics.get("cos_sim", {})
    spearman = cos_sim.get("spearman")
    pearson = cos_sim.get("pearson")
    if spearman is not None:
        wandb.summary[f"{task_name}/cos_sim_spearman"] = spearman
        wandb.summary[f"{task_name}/cos_sim_pearson"] = pearson
        all_spearman[task_name] = spearman

if all_spearman:
    wandb.summary["avg_spearman"] = sum(all_spearman.values()) / len(all_spearman)

wandb.finish()
print(f"Run C URL: {run_c.url}")
```

### Approach D: `wandb.Table` + bar chart

Log a table and let wandb render a bar chart from it. This is the most explicit approach.

```python
run_d = wandb.init(project="LM-STS-CFT", name="logging-test-D-table", job_type="debug", reinit=True)

table = wandb.Table(columns=["task", "cos_sim_spearman", "cos_sim_pearson"])
all_spearman = {}

for task_name, task_data in all_results.items():
    metrics = task_data.get("test", task_data.get("validation", {}))
    cos_sim = metrics.get("cos_sim", {})
    spearman = cos_sim.get("spearman")
    pearson = cos_sim.get("pearson")
    if spearman is not None:
        table.add_data(task_name, spearman, pearson)
        all_spearman[task_name] = spearman

wandb.log({"sts_results": table})

# Also log a bar chart directly
bar_chart = wandb.plot.bar(
    table, "task", "cos_sim_spearman",
    title="STS Cosine Similarity Spearman by Task"
)
wandb.log({"sts_spearman_bar": bar_chart})

if all_spearman:
    wandb.summary["avg_spearman"] = sum(all_spearman.values()) / len(all_spearman)

wandb.finish()
print(f"Run D URL: {run_d.url}")
```

### Approach E: Combined — summary for scalar values + table for overview

Use `wandb.summary` for per-task scalars (shows in run overview), and a `wandb.Table` + bar chart for the visual.

```python
run_e = wandb.init(project="LM-STS-CFT", name="logging-test-E-combined", job_type="debug", reinit=True)

table = wandb.Table(columns=["task", "cos_sim_spearman", "cos_sim_pearson", "eval_time"])
all_spearman = {}

for task_name, task_data in all_results.items():
    metrics = task_data.get("test", task_data.get("validation", {}))
    cos_sim = metrics.get("cos_sim", {})
    spearman = cos_sim.get("spearman")
    pearson = cos_sim.get("pearson")
    eval_time = metrics.get("evaluation_time")
    
    if spearman is not None:
        # Summary: per-task scalars
        wandb.summary[f"{task_name}/cos_sim_spearman"] = spearman
        wandb.summary[f"{task_name}/cos_sim_pearson"] = pearson
        if eval_time is not None:
            wandb.summary[f"{task_name}/eval_time"] = eval_time
        
        # Table row
        table.add_data(task_name, spearman, pearson, eval_time)
        all_spearman[task_name] = spearman

# Aggregate
if all_spearman:
    avg = sum(all_spearman.values()) / len(all_spearman)
    wandb.summary["avg_spearman"] = avg

# Table + bar chart
wandb.log({"sts_results": table})
wandb.log({"sts_spearman_bar": wandb.plot.bar(
    table, "task", "cos_sim_spearman",
    title="STS Cosine Similarity Spearman by Task"
)})

wandb.finish()
print(f"Run E URL: {run_e.url}")
```

## 4. Compare results on the W&B dashboard

Open each run URL above and check:

| Approach | Charts tab | Overview/Summary | Notes |
|----------|-----------|-----------------|-------|
| A (per-step) | Each task gets its own panel with one point | Metrics in summary | Current approach — sparse charts |
| B (single-step) | All metrics logged at step 0 | Metrics in summary | Slightly better — all at same step |
| C (summary-only) | No charts (summary doesn't create history) | All metrics visible | Clean summary, no charts tab |
| D (table) | Bar chart panel | avg in summary | Best visual — explicit bar chart |
| E (combined) | Bar chart panel | Per-task + avg in summary | Best of both worlds |

**Pick the winner and update the eval scripts accordingly.**

```python

```

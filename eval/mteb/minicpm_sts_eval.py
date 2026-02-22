import argparse
import gc
import json
import logging

import torch
import wandb
from model.causal_lm import CausalLMEncoder
from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sts_eval")

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MTEB STS benchmarks with MiniCPM")
    parser.add_argument("--model_path", type=str, default="../../pretrained/sarvam-1",
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default="../../train/output/20260221004650",
                        help="Path to LoRA adapter")
    parser.add_argument("--wandb_project", type=str, default="LM-STS-CFT",
                        help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default="sts-eval",
                        help="W&B run name")
    return parser.parse_args()


def get_split_metrics(task_scores):
    """Try test, then validation split. Returns metrics dict or None."""
    for split in ("test", "validation"):
        if split in task_scores:
            return task_scores[split]
    return None


def main():
    args = parse_args()

    model = CausalLMEncoder(model_path=args.model_path, adapter_path=args.adapter_path)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="eval",
        config={"model_path": args.model_path, "adapter_path": args.adapter_path},
    )

    all_spearman = {}

    for task in TASK_LIST_STS:
        logger.info(f"Running task: {task}")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        results = evaluation.run(model, output_folder="results/minicpm/sts", overwrite_results=True)

        task_scores = results.get(task)
        if task_scores is None:
            logger.error(f"Task '{task}' not found in results. Available keys: {list(results.keys())}")
            continue

        logger.debug(f"Raw results for {task}: {json.dumps(task_scores, indent=2)}")

        metrics = get_split_metrics(task_scores)
        if metrics is None:
            logger.error(f"No test/validation split for '{task}'. Available splits: {list(task_scores.keys())}")
            continue

        cos_sim = metrics.get("cos_sim", {})
        spearman = cos_sim.get("spearman")
        pearson = cos_sim.get("pearson")
        eval_time = metrics.get("evaluation_time")

        log_dict = {}
        if spearman is not None:
            log_dict[f"{task}/cos_sim_spearman"] = spearman
            all_spearman[task] = spearman
        if pearson is not None:
            log_dict[f"{task}/cos_sim_pearson"] = pearson
        if eval_time is not None:
            log_dict[f"{task}/eval_time"] = eval_time
        if log_dict:
            wandb.log(log_dict)

        logger.info(f"{task}: spearman={spearman}, pearson={pearson}")

        # Free fragmented CUDA memory between tasks
        gc.collect()
        torch.cuda.empty_cache()

    if all_spearman:
        avg_spearman = sum(all_spearman.values()) / len(all_spearman)
        wandb.log({"avg_spearman": avg_spearman})
        logger.info(f"Average Spearman: {avg_spearman:.4f}")

        table = wandb.Table(columns=["task", "cos_sim_spearman"])
        for task_name, score in all_spearman.items():
            table.add_data(task_name, score)
        wandb.log({"sts_summary": table})

    wandb.finish()


if __name__ == "__main__":
    main()

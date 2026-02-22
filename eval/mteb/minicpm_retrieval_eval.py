import argparse
import gc
import json
import logging

import torch
import wandb
from model.causal_lm import CausalLMEncoder
from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval_eval")

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

KEY_METRICS = ["ndcg_at_10", "map_at_100", "mrr_at_100", "recall_at_100"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MTEB retrieval benchmarks with MiniCPM")
    parser.add_argument("--model_path", type=str, default="../../pretrained/sarvam-1",
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default="../../train/output/20260221004650",
                        help="Path to LoRA adapter")
    parser.add_argument("--wandb_project", type=str, default="LM-STS-CFT",
                        help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default="retrieval-eval",
                        help="W&B run name")
    return parser.parse_args()


def get_split_metrics(task_scores):
    """Try test, then dev, then validation split. Returns metrics dict or None."""
    for split in ("test", "dev", "validation"):
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

    all_ndcg_at_10 = {}

    for task in TASK_LIST_RETRIEVAL:
        logger.info(f"Running task: {task}")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        results = evaluation.run(model, output_folder="results/minicpm/retrieval", overwrite_results=True)

        task_scores = results.get(task)
        if task_scores is None:
            logger.error(f"Task '{task}' not found in results. Available keys: {list(results.keys())}")
            continue

        logger.debug(f"Raw results for {task}: {json.dumps(task_scores, indent=2)}")

        metrics = get_split_metrics(task_scores)
        if metrics is None:
            logger.error(f"No test/dev/validation split for '{task}'. Available splits: {list(task_scores.keys())}")
            continue

        log_dict = {}
        for metric_name in KEY_METRICS:
            value = metrics.get(metric_name)
            if value is not None:
                log_dict[f"{task}/{metric_name}"] = value

        ndcg = metrics.get("ndcg_at_10")
        if ndcg is not None:
            all_ndcg_at_10[task] = ndcg

        eval_time = metrics.get("evaluation_time")
        if eval_time is not None:
            log_dict[f"{task}/eval_time"] = eval_time

        if log_dict:
            wandb.log(log_dict)

        logger.info(f"{task}: ndcg@10={ndcg}")

        # Free fragmented CUDA memory between tasks
        gc.collect()
        torch.cuda.empty_cache()

    if all_ndcg_at_10:
        avg_ndcg = sum(all_ndcg_at_10.values()) / len(all_ndcg_at_10)
        wandb.log({"avg_ndcg_at_10": avg_ndcg})
        logger.info(f"Average NDCG@10: {avg_ndcg:.4f}")

        table = wandb.Table(columns=["task", "ndcg_at_10"])
        for task_name, score in all_ndcg_at_10.items():
            table.add_data(task_name, score)
        wandb.log({"retrieval_summary": table})

    wandb.finish()


if __name__ == "__main__":
    main()

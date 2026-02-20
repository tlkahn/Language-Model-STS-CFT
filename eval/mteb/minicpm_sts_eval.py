from model.minicpm import MiniCPM
from mteb import MTEB
import logging
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

model_path = '../../pretrained/MiniCPM-2B-dpo-bf16'
adapter_path = '../../train/output/20260220130038'

model = MiniCPM(model_path=model_path,
                adapter_path=adapter_path)

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
    "STSBenchmark"
]

wandb.init(project="LM-STS-CFT", name="sts-eval", job_type="eval")

all_spearman = {}

for task in TASK_LIST_STS:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(tasks=[task], task_langs=["en"])
    results = evaluation.run(model, output_folder=f"results/minicpm/sts", overwrite_results=True)

    task_scores = results.get(task, list(results.values())[0])
    metrics = task_scores.get("test", task_scores.get("validation", {}))
    if metrics:
        cos_sim = metrics.get("cos_sim", {})
        spearman = cos_sim.get("spearman", None)
        pearson = cos_sim.get("pearson", None)
        eval_time = metrics.get("evaluation_time", None)

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

if all_spearman:
    avg_spearman = sum(all_spearman.values()) / len(all_spearman)
    wandb.log({"avg_spearman": avg_spearman})
    logger.info(f"Average Spearman: {avg_spearman:.4f}")

    table = wandb.Table(columns=["task", "cos_sim_spearman"])
    for task_name, score in all_spearman.items():
        table.add_data(task_name, score)
    wandb.log({"sts_summary": table})

wandb.finish()



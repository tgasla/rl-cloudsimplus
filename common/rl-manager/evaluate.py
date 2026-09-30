import json
import os
import re
import time

import numpy as np
import pandas as pd

from utils.evaluation import level_quotas, model_predictor, noop_predictor, play_levels
from utils.levels import EVAL_SPLITS
from utils.misc import get_algorithm, get_suitable_device, source_checkpoint, vectorize_env
from utils.run_dir import STATUS_FILENAME

RULE_CHECKPOINT = "rule"
PROVENANCE = ("arch", "run", "checkpoint", "trained_steps", "trained_on", "finetune", "source",
              "source_run", "seed", "run_status")


def provenance(params) -> dict:
    """Where the evaluated policy came from, read from the run_status.json of every run in its
    transfer chain (benchmark/analyze.py groups and pairs evaluations by these columns).

    arch            the source run's feature_extractor, "default" for SB3's own network (a
                    transfer keeps the network it loads); for a rule-based policy its
                    cloudlet_to_dc_mapping
    run             train_model_dir, the run whose checkpoint is evaluated
    checkpoint      the rule that produced the policy, from the source run to `run`:
                    best_val_model, or best_val_model>model_at_5000 for the model_at_5000
                    checkpoint of a run fine-tuned from the source's best_val_model; "rule" for
                    a rule-based policy
    trained_steps   env steps behind a model_at_<k> checkpoint (its sidecar), else None
    trained_on      the member `run` trained on; finetune is its scope when it is a transfer
    source          the member the chain started from; source_run and seed are that run's
    run_status      "completed" if every run in the chain finished, else the status of the
                    first that did not: a checkpoint of an unfinished chain is not a result
    """
    if params["cloudlet_to_dc_mapping"] != "rl":
        return {**dict.fromkeys(PROVENANCE), "arch": params["cloudlet_to_dc_mapping"],
                "checkpoint": RULE_CHECKPOINT}
    base = params["base_log_dir"]
    run_dir = os.path.normpath(params["train_model_dir"])
    run = _status(base, run_dir)
    rules, root_dir, root = [source_checkpoint(params)], run_dir, run
    statuses = [run["status"]]
    while root.get("mode") == "transfer":
        rules.insert(0, source_checkpoint(root))    # what this transfer loaded from its parent
        root_dir = os.path.normpath(root["train_model_dir"])
        root = _status(base, root_dir)
        statuses.append(root["status"])
    return {"arch": root["feature_extractor"] or "default", "run": run_dir,
            "checkpoint": ">".join(rules),
            "trained_steps": _trained_steps(base, run_dir, rules[-1]),
            "trained_on": run["benchmark_member"],
            "finetune": (run.get("finetune") or "full") if run.get("mode") == "transfer" else None,
            "source": root["benchmark_member"], "source_run": root_dir, "seed": root["seed"],
            "run_status": next((s for s in statuses if s != "completed"), "completed")}


def _status(base_log_dir: str, run_dir: str) -> dict:
    with open(os.path.join(base_log_dir, run_dir, STATUS_FILENAME)) as f:
        return json.load(f)


def _trained_steps(base_log_dir: str, run_dir: str, checkpoint: str) -> int | None:
    """The sidecar SaveAtStepsCallback writes next to model_at_<k>; other checkpoints have none."""
    if not re.fullmatch(r"model_at_\d+", checkpoint):
        return None
    with open(os.path.join(base_log_dir, run_dir, checkpoint + ".json")) as f:
        return json.load(f)["trained_steps"]


def evaluate(params, jobs):
    """Play every level of params["level_split"] once and write evaluation.csv, one row per
    level, stamped with the policy's provenance. An RL policy is the deterministic policy of the
    train_model_dir checkpoint (source_checkpoint); a rule-based cloudlet_to_dc_mapping needs
    no model."""
    split = params["level_split"]
    if split == "train":
        raise ValueError("evaluate needs level_split val, test or lockbox")
    num_cpu = params.get("num_cpu", 16)
    rl = params["cloudlet_to_dc_mapping"] == "rl"
    origin = provenance(params)     # before the JVMs start: a broken chain fails fast
    algorithm = get_algorithm(params["rl_algorithm"], params) if rl else None
    # Each worker's level sampler reads its share of the split from params["num_cpu"]
    # (misc.level_stream), so the env params must state the worker count played here.
    env = vectorize_env(None, algorithm, num_cpu=num_cpu, params={**params, "num_cpu": num_cpu},
                        jobs_json=json.dumps(jobs))

    if rl:
        checkpoint = source_checkpoint(params)
        path = os.path.join(params["base_log_dir"], params["train_model_dir"], checkpoint)
        model = algorithm.load(path, env=env, device=get_suitable_device(params["rl_algorithm"]))
        predict, policy = model_predictor(model), f"{params['train_model_dir']}/{checkpoint}"
    else:
        predict, policy = noop_predictor(params["max_jobs_waiting"]), params["cloudlet_to_dc_mapping"]

    start = time.perf_counter()
    rows = play_levels(env, predict, level_quotas(split, num_cpu))
    elapsed = time.perf_counter() - start
    env.close()

    df = pd.DataFrame(rows).sort_values("level_id").assign(
        member=params["benchmark_member"], split=split, policy=policy, **origin)
    played = df["level_id"].tolist()
    if played != list(EVAL_SPLITS[split]):
        raise RuntimeError(f"played {len(played)} episodes on {len(set(played))} distinct levels, "
                           f"not each of the {len(EVAL_SPLITS[split])} {split} levels once; "
                           f"evaluation.csv not written")
    returns = df["unshaped_return"]
    print(f"{policy} on {params['benchmark_member']}/{split}: {len(df)} levels, unshaped return "
          f"{returns.mean():.4f} ± {returns.std(ddof=1):.4f} (sd), {elapsed:.1f} s")
    if params["log_dir"]:
        df.to_csv(os.path.join(params["log_dir"], "evaluation.csv"), index=False)
    return df

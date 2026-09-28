import json
import os
import time

import numpy as np
import pandas as pd

from utils.evaluation import level_quotas, model_predictor, noop_predictor, play_levels
from utils.misc import get_algorithm, get_suitable_device, source_checkpoint, vectorize_env


def evaluate(params, jobs):
    """Play every level of params["level_split"] once and write evaluation.csv, one row per
    level. An RL policy is the deterministic policy of the train_model_dir checkpoint
    (source_checkpoint); a rule-based cloudlet_to_dc_mapping needs no model."""
    split = params["level_split"]
    if split == "train":
        raise ValueError("evaluate needs level_split val, test or lockbox")
    num_cpu = params.get("num_cpu", 16)
    rl = params["cloudlet_to_dc_mapping"] == "rl"
    algorithm = get_algorithm(params["rl_algorithm"], params) if rl else None
    env = vectorize_env(None, algorithm, num_cpu=num_cpu, params=params, jobs_json=json.dumps(jobs))

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
        member=params["benchmark_member"], split=split, policy=policy)
    returns = df["unshaped_return"]
    print(f"{policy} on {params['benchmark_member']}/{split}: {len(df)} levels, unshaped return "
          f"{returns.mean():.4f} ± {returns.std(ddof=1):.4f} (sd), {elapsed:.1f} s")
    if params["log_dir"]:
        df.to_csv(os.path.join(params["log_dir"], "evaluation.csv"), index=False)
    return df

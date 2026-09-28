"""Deterministic evaluation on the levels of a split: every level played exactly once.

Shared by `mode: evaluate` and the best-on-val checkpoint callback. The benchmark metric is
the unshaped return (the sum of info["unshaped_reward"]); the shaped return is kept only as
a diagnostic.
"""
import copy

import numpy as np

from utils.levels import eval_levels

EPISODE_SUMS = ("sla_value_realized", "sla_penalty_paid", "resource_cost", "jobs_met",
                "jobs_violated", "jobs_expired_unplaced")


def eval_env_params(params: dict, split: str, num_workers: int, port_offset: int = 0) -> dict:
    """Params for a vectorised env whose workers enumerate `split`. log_dir is dropped so it
    writes no monitor file of its own; port_offset keeps its JVMs clear of a training env's."""
    out = copy.deepcopy(params)
    out.update(level_split=split, num_cpu=num_workers, log_dir=None,
               grpc_base_port=params.get("grpc_base_port", 50051) + port_offset)
    return out


def level_quotas(split: str, num_workers: int) -> list[int]:
    return [len(eval_levels(split, rank, num_workers)) for rank in range(num_workers)]


def play_levels(vec_env, predict, quotas: list[int]) -> list[dict]:
    """Play `quotas[i]` episodes on worker i and return one row per episode.

    A worker that has finished its share keeps stepping with the others, but its extra
    episodes are not recorded. `predict(obs, action_masks)` returns the batch of actions.
    """
    n = vec_env.num_envs
    finished = [0] * n
    running = [dict.fromkeys(("unshaped_return", "shaped_return", "steps") + EPISODE_SUMS, 0.0)
               for _ in range(n)]
    rows = []
    obs = vec_env.reset()
    while any(finished[i] < quotas[i] for i in range(n)):
        masks = np.stack(vec_env.env_method("action_masks"))
        obs, rewards, dones, infos = vec_env.step(predict(obs, masks))
        for i in range(n):
            if finished[i] >= quotas[i]:
                continue
            acc, info = running[i], infos[i]
            acc["shaped_return"] += float(rewards[i])
            acc["unshaped_return"] += info["unshaped_reward"]
            acc["steps"] += 1
            for key in EPISODE_SUMS:
                acc[key] += info[key]
            if dones[i]:
                rows.append(dict(acc, level_id=info["level_id"], worker=i,
                                 offered_value=info["offered_value"],
                                 terminated=not info.get("TimeLimit.truncated", False)))
                running[i] = dict.fromkeys(acc, 0.0)
                finished[i] += 1
    return rows


def model_predictor(model):
    def predict(obs, masks):
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        return actions
    return predict


def noop_predictor(action_dim: int):
    """For rule-based policies: Java decides and ignores the action."""
    def predict(obs, masks):
        return np.zeros((len(masks), action_dim), dtype=np.int64)
    return predict

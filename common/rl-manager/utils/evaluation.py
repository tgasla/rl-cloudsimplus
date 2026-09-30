"""Deterministic evaluation on the levels of a split: every level played exactly once.

Shared by `mode: evaluate` and the best-on-val checkpoint callback. The benchmark metric is
the unshaped return (the sum of info["unshaped_reward"]); the shaped return is kept only as
a diagnostic.
"""
import copy

import numpy as np
import torch

from extractors.pointer_policy import TokenHeadPolicy
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
    episodes are not recorded. `predict(obs, action_masks)` returns the batch of actions; a
    predictor with a `tie_breaks` attribute (model_predictor) adds its per-episode sum.
    """
    n = vec_env.num_envs
    finished = [0] * n
    counts_ties = hasattr(predict, "tie_breaks")
    keys = ("unshaped_return", "shaped_return", "steps") + EPISODE_SUMS
    running = [dict.fromkeys(keys + (("tie_breaks",) if counts_ties else ()), 0.0)
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
            if counts_ties:
                acc["tie_breaks"] += int(predict.tie_breaks[i])
            if dones[i]:
                rows.append(dict(acc, level_id=info["level_id"], worker=i,
                                 offered_value=info["offered_value"],
                                 terminated=not info.get("TimeLimit.truncated", False)))
                running[i] = dict.fromkeys(acc, 0.0)
                finished[i] += 1
    return rows


def _action_logits(policy, obs) -> torch.Tensor:
    """The raw logits [B, J * D] of a token head or of SB3's maskable policy."""
    if isinstance(policy, TokenHeadPolicy):
        return policy._logits(policy._tokens(obs))
    features = policy.extract_features(obs, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def model_predictor(model, vec_env):
    """The deterministic policy of `model` on `vec_env`: for each job slot, the legal action with
    the highest logit. Exact ties go to the no-op, then to the DC whose name sorts first
    (params["datacenters"][k - 1]["name"] for action k), never to the lower action index: a
    pointer head scores DCs in identical states exactly alike, and the index would make a
    relabelled topology play differently. A strictly best action is never changed. The raw
    logits are compared, not the log-probabilities: normalising them rounds logits 1e-7 apart
    into false ties, and does so differently under another DC numbering.
    On the CPU the forward pass runs on one thread: with many threads the kernels round some
    rows of a batch differently, and DCs in identical states then score a few ulps apart.
    After each call, predict.tie_breaks holds per worker the decisions the tie rule settled."""
    rank = None

    def predict(obs, masks):
        nonlocal rank
        policy = model.policy
        if rank is None:
            names = [dc["name"] for dc in vec_env.get_attr("params", [0])[0]["datacenters"]]
            # the no-op, then the DCs by name; the actions past the real DCs are always masked
            rank = np.arange(int(policy.action_space.nvec[0]))
            rank[1:len(names) + 1] = 1 + np.argsort(np.argsort(names))
        policy.set_training_mode(False)
        obs_tensor, _ = policy.obs_to_tensor(obs)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.no_grad():
                logits = _action_logits(policy, obs_tensor).cpu().numpy()
        finally:
            torch.set_num_threads(threads)
        legal = np.asarray(masks, dtype=bool).reshape(len(logits), -1, len(rank))    # [B, J, D]
        scores = logits.reshape(legal.shape)
        best = np.where(legal, scores, -np.inf).max(axis=-1, keepdims=True)
        tied = legal & (scores == best)
        predict.tie_breaks = (tied.sum(axis=-1) > 1).sum(axis=-1)
        return np.where(tied, rank, len(rank)).argmin(axis=-1)

    predict.tie_breaks = None
    return predict


def noop_predictor(action_dim: int):
    """For rule-based policies: Java decides and ignores the action."""
    def predict(obs, masks):
        return np.zeros((len(masks), action_dim), dtype=np.int64)
    return predict

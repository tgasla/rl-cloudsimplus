"""Shared input featurisation for the set and graph extractors.

Observation layout (JobPlacementEnv):
  infrastructure_state  per host slot [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts]
  jobs_waiting_state    per job slot  [cores, nominal_runtime_ref, time_to_due, s0, s1, s2]
  reach_mask            [job slot, action] legality; action k places on the DC with dc_id k

Every set/graph extractor featurises the same way, so comparing them compares architectures,
not input scaling:
  - dc_type is one-hot (1=cloud, 2=edge, 3=micro; 0 marks a padding host slot), never a raw
    scalar, which a linear layer would read as an ordering of tiers
  - magnitudes go through log1p, so counts that grow with topology size stay in range
  - dc_id is only a grouping index and is never embedded; DC slot k is action k
"""

import torch
import torch.nn.functional as F

HOST_FEATURES = 5
JOB_FEATURES = 6
N_DC_TYPES = 3
IDX_DC_ID, IDX_DC_TYPE, IDX_VM_CAP, IDX_FREE_PES, IDX_BACKLOG = range(HOST_FEATURES)

HOST_INPUT_DIM = N_DC_TYPES + 3  # one-hot type, log1p(vm_capacity, free, backlog)
JOB_INPUT_DIM = JOB_FEATURES     # log1p(cores, runtime, time_to_due), sensitivity one-hot
DC_INPUT_DIM = N_DC_TYPES + 5    # one-hot type, log1p(capacity, free, max host free, backlog, hosts)


def split_observation(observations: dict, device: torch.device):
    """Return hosts [B, S, HOST_FEATURES], jobs [B, J, JOB_FEATURES], reach [B, J, D] as floats."""
    infr = observations["infrastructure_state"].to(device).float()
    jobs = observations["jobs_waiting_state"].to(device).float()
    reach = observations["reach_mask"].to(device).float()
    batch = infr.shape[0]
    jobs = jobs.view(batch, -1, JOB_FEATURES)
    return infr.view(batch, -1, HOST_FEATURES), jobs, reach.view(batch, jobs.shape[1], -1)


def host_inputs(hosts: torch.Tensor):
    """Return dc_ids [B, S] (long), host_mask [B, S], per-host inputs [B, S, HOST_INPUT_DIM]."""
    dc_ids = hosts[..., IDX_DC_ID].long()
    dc_type = F.one_hot(hosts[..., IDX_DC_TYPE].long(), N_DC_TYPES + 1)[..., 1:].float()
    magnitudes = torch.log1p(hosts[..., [IDX_VM_CAP, IDX_FREE_PES, IDX_BACKLOG]])
    return dc_ids, dc_ids > 0, torch.cat([dc_type, magnitudes], dim=-1)


def job_inputs(jobs: torch.Tensor):
    """Return job_mask [B, J] (padding slots have 0 cores) and inputs [B, J, JOB_INPUT_DIM]."""
    return jobs[..., 0] > 0, torch.cat([torch.log1p(jobs[..., :3]), jobs[..., 3:]], dim=-1)


def dc_inputs(hosts: torch.Tensor, n_slots: int):
    """Aggregate host rows into DC tokens by dc_id value, independent of host row order.

    Returns dc_mask [B, n_slots] and inputs [B, n_slots, DC_INPUT_DIM]. Slot k holds the DC
    with dc_id k, so it lines up with action k and with reach column k. Slot 0 collects the
    padding host rows and is always masked out.
    """
    batch = hosts.shape[0]
    dc_ids = hosts[..., IDX_DC_ID].long()
    dc_type = F.one_hot(hosts[..., IDX_DC_TYPE].long(), N_DC_TYPES + 1)[..., 1:].float()

    def scatter(values: torch.Tensor, reduce: str) -> torch.Tensor:
        out = torch.zeros(batch, n_slots, *values.shape[2:], device=hosts.device)
        index = dc_ids.view(batch, -1, *([1] * (values.dim() - 2))).expand_as(values)
        return out.scatter_reduce(1, index, values, reduce=reduce, include_self=True)

    host_count = scatter(torch.ones_like(hosts[..., IDX_DC_ID]), "sum")
    totals = scatter(hosts[..., [IDX_VM_CAP, IDX_FREE_PES, IDX_BACKLOG]], "sum")
    max_free = scatter(hosts[..., IDX_FREE_PES], "amax")
    features = torch.cat([
        scatter(dc_type, "amax"),                       # every host of a DC has the same type
        torch.log1p(totals[..., :2]),                   # capacity, free
        torch.log1p(max_free).unsqueeze(-1),            # a job needs its PEs on one host
        torch.log1p(totals[..., 2:]),                   # backlog
        torch.log1p(host_count).unsqueeze(-1),
    ], dim=-1)
    dc_mask = host_count > 0
    dc_mask[:, 0] = False
    return dc_mask, features


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim 1 of the real elements only, so padding count cannot shift the result."""
    weights = mask.unsqueeze(-1).float()
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

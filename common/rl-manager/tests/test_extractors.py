"""Feature extractors against the current observation layout.

The set/graph extractors must not depend on how DCs are numbered or ordered, nor on the
order of job slots: relabelling DCs (host rows, dc_id values and reach columns together)
or permuting job slots (job rows and reach rows together) must leave their output unchanged.
"""

import os

import numpy as np
import pytest
import torch

from conftest import SPEC_SHAPE, ring_topology

N_RING = 7
INVARIANT = ["deepsets", "turret"]


@pytest.fixture
def obs_space(make_env):
    return make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE).observation_space


def _random_obs(space, rng, batch=4):
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    types = [1] + [2, 3] * N_RING  # cloud, then edge/micro alternating
    obs = {key: np.zeros((batch,) + s.shape, dtype=s.dtype) for key, s in space.spaces.items()}
    for b in range(batch):
        hosts = []
        for dc_id in range(1, N_RING + 2):
            cap = {1: 32, 2: 16, 3: 8}[types[dc_id - 1]]
            for _ in range(rng.integers(1, 4)):
                hosts.append([dc_id, types[dc_id - 1], cap, rng.integers(0, cap + 1),
                              rng.integers(0, 200)])
        obs["infrastructure_state"][b, :len(hosts) * 5] = np.ravel(hosts)
        n_real = rng.integers(1, n_jobs)
        jobs = np.zeros((n_jobs, 6), dtype=np.int32)
        jobs[:n_real, 0] = rng.integers(1, 9, n_real)
        jobs[:n_real, 1] = rng.integers(3, 61, n_real)
        jobs[:n_real, 2] = rng.integers(0, 90, n_real)
        jobs[np.arange(n_real), 3 + rng.integers(0, 3, n_real)] = 1
        obs["jobs_waiting_state"][b] = jobs.ravel()
        reach = np.zeros((n_jobs, n_slots), dtype=np.int8)
        reach[:, 0] = 1
        reach[:n_real, 1:N_RING + 2] = rng.integers(0, 2, (n_real, N_RING + 1))
        obs["reach_mask"][b] = reach.ravel()
    return obs


def _relabel(obs, rng):
    """Renumber the real DCs, shuffle host rows, and permute job slots, consistently."""
    out = {key: value.copy() for key, value in obs.items()}
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    for b in range(len(obs["infrastructure_state"])):
        new_id = np.arange(n_slots)
        new_id[1:N_RING + 2] = rng.permutation(np.arange(1, N_RING + 2))
        hosts = out["infrastructure_state"][b].reshape(-1, 5)
        hosts[:, 0] = new_id[hosts[:, 0]]
        out["infrastructure_state"][b] = hosts[rng.permutation(len(hosts))].ravel()

        reach = obs["reach_mask"][b].reshape(n_jobs, n_slots)
        relabelled = np.zeros_like(reach)
        relabelled[:, new_id] = reach
        order = rng.permutation(n_jobs)
        out["reach_mask"][b] = relabelled[order].ravel()
        out["jobs_waiting_state"][b] = obs["jobs_waiting_state"][b].reshape(n_jobs, 6)[order].ravel()
    return out


def _extractor(name, space):
    from extractors import build_extractor_kwargs, get_extractor_class

    torch.manual_seed(0)
    model = get_extractor_class(name)(space, **build_extractor_kwargs(name, SPEC_SHAPE))
    return model.eval()


def _features(model, obs):
    with torch.no_grad():
        return model({key: torch.as_tensor(value) for key, value in obs.items()})


@pytest.mark.parametrize("name", ["euromlsys"] + INVARIANT)
def test_extractor_reads_the_observation(name, obs_space):
    obs = _random_obs(obs_space, np.random.default_rng(0))
    features = _features(_extractor(name, obs_space), obs)
    assert features.shape == (4, 64) and torch.isfinite(features).all()


@pytest.mark.parametrize("name", INVARIANT)
def test_set_extractors_ignore_dc_numbering_and_slot_order(name, obs_space):
    rng = np.random.default_rng(1)
    obs = _random_obs(obs_space, rng)
    model = _extractor(name, obs_space)
    torch.testing.assert_close(_features(model, obs), _features(model, _relabel(obs, rng)),
                               atol=1e-4, rtol=1e-4)


def test_flat_mlp_is_not_invariant(obs_space):
    # Control: if the relabelling were a no-op, the invariance test above would prove nothing.
    rng = np.random.default_rng(1)
    obs = _random_obs(obs_space, rng)
    model = _extractor("euromlsys", obs_space)
    assert not torch.allclose(_features(model, obs), _features(model, _relabel(obs, rng)))


def test_extractors_written_for_the_old_layout_are_refused():
    from extractors import get_extractor_class

    with pytest.raises(ValueError, match="port it first"):
        get_extractor_class("fusion")


def test_set_extractors_never_embed_dc_ids():
    # dc_id is a grouping index; an nn.Embedding over it cannot transfer to unseen DC counts.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extractors")
    for file_name in ["featurize.py", "deepsets_extractor.py", "turret_extractor.py"]:
        assert "Embedding(" not in open(os.path.join(here, file_name)).read(), file_name


def _fill_padding(obs, rng):
    """Arbitrary values in every padding slot's non-identifying columns."""
    out = {key: value.copy() for key, value in obs.items()}
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    for b in range(len(obs["infrastructure_state"])):
        hosts = out["infrastructure_state"][b].reshape(-1, 5)
        pad = hosts[:, 0] == 0
        hosts[pad, 1] = rng.integers(0, 4, pad.sum())
        hosts[pad, 2:] = rng.integers(0, 50, (pad.sum(), 3))
        jobs = out["jobs_waiting_state"][b].reshape(n_jobs, 6)
        reach = out["reach_mask"][b].reshape(n_jobs, n_slots)
        pad = jobs[:, 0] == 0
        jobs[pad, 1:] = rng.integers(0, 50, (pad.sum(), 5))
        reach[pad] = rng.integers(0, 2, (pad.sum(), n_slots))
    return out


@pytest.mark.parametrize("name", INVARIANT)
def test_padding_slots_are_ignored(name, obs_space):
    rng = np.random.default_rng(2)
    obs = _random_obs(obs_space, rng)
    model = _extractor(name, obs_space)
    torch.testing.assert_close(_features(model, obs), _features(model, _fill_padding(obs, rng)),
                               atol=1e-5, rtol=1e-5)


def test_deepsets_features_do_not_move_with_the_number_of_dcs(obs_space):
    # Every real DC duplicated under an unused dc_id, reach columns included: a masked mean
    # cannot tell the difference, an unmasked one (checklist red flag 3) would.
    rng = np.random.default_rng(3)
    obs = _random_obs(obs_space, rng)
    doubled = {key: value.copy() for key, value in obs.items()}
    n_jobs, n_slots, n_real = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"], N_RING + 1
    for b in range(len(obs["infrastructure_state"])):
        hosts = obs["infrastructure_state"][b].reshape(-1, 5)
        real = hosts[hosts[:, 0] > 0]
        copies = real.copy()
        copies[:, 0] += n_real
        rows = np.concatenate([real, copies])
        doubled["infrastructure_state"][b] = 0
        doubled["infrastructure_state"][b, :rows.size] = rows.ravel()
        reach = doubled["reach_mask"][b].reshape(n_jobs, n_slots)
        reach[:, n_real + 1:2 * n_real + 1] = reach[:, 1:n_real + 1]
    model = _extractor("deepsets", obs_space)
    torch.testing.assert_close(_features(model, obs), _features(model, doubled), atol=1e-5, rtol=1e-5)


def test_deepsets_features_depend_on_where_jobs_may_go(obs_space):
    rng = np.random.default_rng(4)
    obs = _random_obs(obs_space, rng)
    moved = {key: value.copy() for key, value in obs.items()}
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    reach = moved["reach_mask"].reshape(-1, n_jobs, n_slots)
    reach[:, :, 1:N_RING + 2] = 1 - reach[:, :, 1:N_RING + 2]
    reach[moved["jobs_waiting_state"].reshape(-1, n_jobs, 6)[:, :, 0] == 0] = 0
    reach[:, :, 0] = 1
    model = _extractor("deepsets", obs_space)
    assert not torch.allclose(_features(model, obs), _features(model, moved))


def test_turret_graph_links_hosts_of_a_dc_and_jobs_to_the_hosts_they_may_use():
    from extractors.turret_extractor import TurretGNNExtractor

    # Hosts 0,1 in DC 1, host 2 in DC 2, host 3 padding; job 0 may use DC 1, job 1 DC 2,
    # job 2 is padding (its reach row must not matter). Nodes: hosts 0-3, jobs 4-6.
    dc_ids = torch.tensor([[1, 1, 2, 0]])
    host_mask = dc_ids > 0
    job_mask = torch.tensor([[True, True, False]])
    reach = torch.zeros(1, 3, 3)
    reach[0, :, 0] = 1
    reach[0, 0, 1] = reach[0, 1, 2] = 1
    reach[0, 2, 1:] = 1
    edges = TurretGNNExtractor._edges(dc_ids, host_mask, job_mask, reach)
    got = sorted(map(tuple, edges.t().tolist()))
    expected = sorted([(0, 1), (1, 0), (4, 0), (4, 1), (5, 2), (0, 4), (1, 4), (2, 5)])
    assert got == expected


# ─── Pointer head (A5, A3 SPANE) ─────────────────────────────────────────────

def _policy(name, env):
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
    from extractors import (build_extractor_kwargs, build_policy_head_kwargs,
                            get_extractor_class, get_policy_class)

    torch.manual_seed(0)
    cls = get_policy_class(name, MaskableMultiInputActorCriticPolicy)
    return cls(env.observation_space, env.action_space, lambda _: 3e-4,
               features_extractor_class=get_extractor_class(name),
               features_extractor_kwargs=build_extractor_kwargs(name, SPEC_SHAPE),
               **build_policy_head_kwargs(name, SPEC_SHAPE)).eval()


def _relabel_with_maps(obs, rng):
    """_relabel, also returning the DC id map and job slot order per batch element."""
    out = {key: value.copy() for key, value in obs.items()}
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    maps = []
    for b in range(len(obs["infrastructure_state"])):
        new_id = np.arange(n_slots)
        new_id[1:N_RING + 2] = rng.permutation(np.arange(1, N_RING + 2))
        hosts = out["infrastructure_state"][b].reshape(-1, 5)
        hosts[:, 0] = new_id[hosts[:, 0]]
        out["infrastructure_state"][b] = hosts[rng.permutation(len(hosts))].ravel()
        reach = obs["reach_mask"][b].reshape(n_jobs, n_slots)
        relabelled = np.zeros_like(reach)
        relabelled[:, new_id] = reach
        order = rng.permutation(n_jobs)
        out["reach_mask"][b] = relabelled[order].ravel()
        out["jobs_waiting_state"][b] = obs["jobs_waiting_state"][b].reshape(n_jobs, 6)[order].ravel()
        maps.append((new_id, order))
    return out, maps


def _probs(policy, obs):
    masks = obs["reach_mask"].astype(bool)
    with torch.no_grad():
        dist = policy.get_distribution({k: torch.as_tensor(v) for k, v in obs.items()},
                                       action_masks=masks)
    return torch.stack([d.probs for d in dist.distributions], dim=1).numpy()  # [B, J, D]


@pytest.mark.parametrize("name", ["a5", "spane", "a5_no_cross_attention"])
def test_pointer_head_permutes_its_policy_exactly_with_dcs_and_jobs(name, make_env):
    # The lemma behind Delta_relabel = 0: renumbering DCs and reordering job slots permutes
    # the action distribution, so a relabelled topology cannot change what the policy does.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(5)
    obs = _random_obs(env.observation_space, rng)
    relabelled, maps = _relabel_with_maps(obs, rng)
    policy = _policy(name, env)
    p, q = _probs(policy, obs), _probs(policy, relabelled)
    for b, (new_id, order) in enumerate(maps):
        # relabelled slot i holds original job order[i]; action new_id[k] is original action k
        np.testing.assert_allclose(q[b][:, new_id], p[b][order], atol=1e-5)
    with torch.no_grad():
        values = [policy.predict_values({k: torch.as_tensor(v) for k, v in o.items()})
                  for o in (obs, relabelled)]
    torch.testing.assert_close(values[0], values[1], atol=1e-5, rtol=1e-5)


def test_positional_head_is_not_equivariant(make_env):
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(5)
    obs = _random_obs(env.observation_space, rng)
    relabelled, maps = _relabel_with_maps(obs, rng)
    for name in ("euromlsys", "a5_positional_head"):
        policy = _policy(name, env)
        p, q = _probs(policy, obs), _probs(policy, relabelled)
        new_id, order = maps[0]
        assert not np.allclose(q[0][:, new_id], p[0][order], atol=1e-3), name


@pytest.mark.parametrize("name", ["a5", "deepsets", "turret"])
def test_finetune_scope_trains_exactly_the_chosen_part(name, make_env):
    from utils.misc import apply_finetune_scope

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)

    class Model:
        policy = _policy(name, env)

    def trainable(model):
        return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    model = Model()
    total = trainable(model)
    extractor = sum(p.numel() for p in model.policy.features_extractor.parameters())
    assert 0 < extractor < total
    apply_finetune_scope(model, "head")
    assert trainable(model) == total - extractor
    apply_finetune_scope(model, "extractor")
    assert trainable(model) == extractor
    with pytest.raises(ValueError):
        apply_finetune_scope(model, "encoder")


def test_pointer_policy_survives_save_and_load(make_env, tmp_path):
    from sb3_contrib import MaskablePPO
    from extractors import build_extractor_kwargs, build_policy_head_kwargs, get_extractor_class
    from extractors.pointer_policy import PointerPolicy

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    model = MaskablePPO(PointerPolicy, env, device="cpu", seed=0, policy_kwargs=dict(
        features_extractor_class=get_extractor_class("spane"),
        features_extractor_kwargs=build_extractor_kwargs("spane", SPEC_SHAPE),
        **build_policy_head_kwargs("spane", SPEC_SHAPE)))
    model.save(tmp_path / "m")
    loaded = MaskablePPO.load(tmp_path / "m", device="cpu")
    assert isinstance(loaded.policy, PointerPolicy) and loaded.policy.reach_input is False
    obs = _random_obs(env.observation_space, np.random.default_rng(6), batch=2)
    np.testing.assert_allclose(_probs(model.policy.eval(), obs), _probs(loaded.policy.eval(), obs))


def test_turret_head_is_equivariant_in_jobs_and_positional_in_dcs(make_env):
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(7)
    obs = _random_obs(env.observation_space, rng)
    policy = _policy("turret", env)
    n_jobs = SPEC_SHAPE["max_jobs_waiting"]

    # Job slots permuted, DCs untouched: the per-job rows permute with them.
    order = rng.permutation(n_jobs)
    jobs_only = {key: value.copy() for key, value in obs.items()}
    for key, width in (("jobs_waiting_state", 6), ("reach_mask", SPEC_SHAPE["max_datacenters"])):
        rows = obs[key].reshape(-1, n_jobs, width)
        jobs_only[key] = rows[:, order].reshape(obs[key].shape)
    np.testing.assert_allclose(_probs(policy, jobs_only), _probs(policy, obs)[:, order], atol=1e-5)

    relabelled, maps = _relabel_with_maps(obs, rng)
    p, q = _probs(policy, obs), _probs(policy, relabelled)
    new_id, order = maps[0]
    assert not np.allclose(q[0][:, new_id], p[0][order], atol=1e-3)


def test_turret_policy_survives_save_and_load(make_env, tmp_path):
    from sb3_contrib import MaskablePPO
    from extractors import build_extractor_kwargs, build_policy_head_kwargs, get_extractor_class
    from extractors.pointer_policy import PerJobPolicy

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    model = MaskablePPO(PerJobPolicy, env, device="cpu", seed=0, policy_kwargs=dict(
        features_extractor_class=get_extractor_class("turret"),
        features_extractor_kwargs=build_extractor_kwargs("turret", SPEC_SHAPE),
        **build_policy_head_kwargs("turret", SPEC_SHAPE)))
    model.save(tmp_path / "m")
    loaded = MaskablePPO.load(tmp_path / "m", device="cpu")
    assert isinstance(loaded.policy, PerJobPolicy)
    obs = _random_obs(env.observation_space, np.random.default_rng(8), batch=2)
    np.testing.assert_allclose(_probs(model.policy.eval(), obs), _probs(loaded.policy.eval(), obs),
                               atol=1e-6)

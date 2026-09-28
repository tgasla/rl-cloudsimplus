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
        get_extractor_class("spane")


def test_set_extractors_never_embed_dc_ids():
    # dc_id is a grouping index; an nn.Embedding over it cannot transfer to unseen DC counts.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extractors")
    for file_name in ["featurize.py", "deepsets_extractor.py", "turret_extractor.py"]:
        assert "Embedding(" not in open(os.path.join(here, file_name)).read(), file_name

"""Feature extractors against the current observation layout.

The set/graph extractors must not depend on how DCs are numbered or ordered, nor on the
order of job slots: relabelling DCs (host rows, dc_id values and reach columns together)
or permuting job slots (job rows and reach rows together) must leave their output unchanged.
"""

import os

import numpy as np
import pytest
import torch

from conftest import SPEC_SHAPE, ring_member, ring_topology

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


def _with_dcs_doubled(obs):
    """Every real DC duplicated under an unused dc_id, reach columns included."""
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
    return doubled


def test_deepsets_features_do_not_move_with_the_number_of_dcs(obs_space):
    # A masked mean cannot tell the doubled DCs apart, an unmasked one (checklist red flag 3) would.
    obs = _random_obs(obs_space, np.random.default_rng(3))
    model = _extractor("deepsets", obs_space)
    torch.testing.assert_close(_features(model, obs), _features(model, _with_dcs_doubled(obs)),
                               atol=1e-5, rtol=1e-5)


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


def test_turret_reads_the_graph_out_with_a_set_transformer(obs_space):
    # F_read(H) = 1/K sum_k [DECODER(ENCODER(H))]_k (Yang et al., AAAI-24): Buterez et al.'s
    # set-transformer readout, whose encoder lets every node attend to every other node before
    # the seed points pool them, and whose K seed outputs are averaged.
    from torch_geometric.nn.aggr import SetTransformerAggregation

    model = _extractor("turret", obs_space)
    read = model.set_transformer
    assert isinstance(read, SetTransformerAggregation) and len(read.encoders) > 0 and not read.concat
    obs = _random_obs(obs_space, np.random.default_rng(8))
    before = _features(model, obs)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in read.encoders.parameters():
            p.add_(torch.randn_like(p))
    assert not torch.allclose(_features(model, obs), before, atol=1e-4)


# ─── Policies: token heads (A3, A5, V1-V5) and TURRET's output network (A4) ───

# The TokenEncoder architectures: SPANE (A3), A5 and A5's ablations V1-V5.
TOKEN_ARCHS = ["spane", "a5", "a5_positional_head", "a5_unmasked_pool", "a5_scalar_dc_type",
               "a5_dc_id_embedding", "a5_no_reach"]


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


def _model(name, env):
    """A MaskablePPO on architecture `name`, as train.py builds it."""
    from sb3_contrib import MaskablePPO
    from extractors import (build_extractor_kwargs, build_policy_head_kwargs,
                            get_extractor_class, get_policy_class)

    return MaskablePPO(get_policy_class(name, "MultiInputPolicy"), env, n_steps=8, batch_size=8,
                       device="cpu", seed=0, policy_kwargs=dict(
                           features_extractor_class=get_extractor_class(name),
                           features_extractor_kwargs=build_extractor_kwargs(name, SPEC_SHAPE),
                           **build_policy_head_kwargs(name, SPEC_SHAPE)))


def _tensors(obs):
    return {key: torch.as_tensor(value) for key, value in obs.items()}


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
        dist = policy.get_distribution(_tensors(obs), action_masks=masks)
    return torch.stack([d.probs for d in dist.distributions], dim=1).numpy()  # [B, J, D]


def _logits(policy, obs):
    """Unmasked per-slot log-probabilities [B, J, D]."""
    with torch.no_grad():
        dist = policy.get_distribution(_tensors(obs))
    return torch.stack([d.logits for d in dist.distributions], dim=1).numpy()


@pytest.mark.parametrize("name", ["spane", "a5", "a5_no_reach"])
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
        values = [policy.predict_values(_tensors(o)) for o in (obs, relabelled)]
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


@pytest.mark.parametrize("name", TOKEN_ARCHS)
def test_token_policies_ignore_padding_slots(name, make_env):
    # What padding slots hold must reach neither a real job's scores nor the value; V2
    # included: it unmasks the pool over DC slots, not the padding hosts or the job slots.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(2)
    obs = _random_obs(env.observation_space, rng)
    filled = _fill_padding(obs, rng)
    policy = _policy(name, env)
    real = obs["jobs_waiting_state"].reshape(len(obs["jobs_waiting_state"]), -1, 6)[:, :, 0] > 0
    np.testing.assert_allclose(_logits(policy, filled)[real], _logits(policy, obs)[real], atol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(policy.predict_values(_tensors(filled)),
                                   policy.predict_values(_tensors(obs)), atol=1e-5, rtol=1e-5)


def test_a3_and_a5_share_one_head(make_env):
    # Every legal pair has reach 1 under the action mask, so a reach term in the head would
    # only add a constant to the legal pairs' scores: reach enters A5 through its encoder, and
    # the heads of A3 and A5 are the same network.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    extractor = ("features_extractor.", "pi_features_extractor.", "vf_features_extractor.")
    heads = [{key: value.shape for key, value in _policy(name, env).state_dict().items()
              if not key.startswith(extractor)} for name in ("spane", "a5")]
    assert heads[0] == heads[1]


# What each variant changes in A5's encoder; V1 keeps the encoder and changes the head.
A5_SWITCHES = {
    "spane": {"cross_attention": False},                # A3
    "a5_positional_head": {},                           # V1
    "a5_unmasked_pool": {"unmasked_pool": True},        # V2
    "a5_scalar_dc_type": {"scalar_dc_type": True},      # V3
    "a5_dc_id_embedding": {"dc_id_embedding": True},    # V4
    "a5_no_reach": {"no_reach": True},                  # V5
}


@pytest.mark.parametrize("name", list(A5_SWITCHES))
def test_each_variant_turns_on_its_own_switch_only(name):
    from extractors import build_extractor_kwargs, get_policy_class
    from extractors.pointer_policy import PointerPolicy, PositionalHeadPolicy

    a5 = build_extractor_kwargs("a5", SPEC_SHAPE)
    kwargs = build_extractor_kwargs(name, SPEC_SHAPE)
    assert {key: value for key, value in kwargs.items() if a5.get(key, False) != value} \
        == A5_SWITCHES[name]
    head = PositionalHeadPolicy if name == "a5_positional_head" else PointerPolicy
    assert get_policy_class(name, None) is head and get_policy_class("a5", None) is PointerPolicy


def test_a5_without_cross_attention_is_not_registered_as_an_ablation():
    # It would be A3: SPANE's encoder under the same pointer head.
    from extractors import get_extractor_class

    with pytest.raises(ValueError, match="Unknown feature extractor"):
        get_extractor_class("a5_no_cross_attention")


def _encoder_tokens(name, env, obs):
    with torch.no_grad():
        tokens = _policy(name, env).features_extractor.tokens(_tensors(obs))
    return tokens.dc.numpy(), tokens.job.numpy()


def test_a3_v3_and_v5_differ_from_a5_where_they_claim_to(make_env):
    from extractors.featurize import DC_INPUT_DIM, N_DC_TYPES

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    obs = _random_obs(env.observation_space, np.random.default_rng(12))
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    real = obs["jobs_waiting_state"].reshape(-1, n_jobs, 6)[:, :, 0] > 0
    rerouted = {key: value.copy() for key, value in obs.items()}      # other DCs within reach
    reach = rerouted["reach_mask"].reshape(-1, n_jobs, n_slots)
    reach[:, :, 1:N_RING + 2] = np.where(real[..., None], 1 - reach[:, :, 1:N_RING + 2], 0)
    one_less = {key: value.copy() for key, value in obs.items()}      # job 0 has left
    one_less["jobs_waiting_state"].reshape(-1, n_jobs, 6)[:, 0] = 0
    one_less["reach_mask"].reshape(-1, n_jobs, n_slots)[:, 0, 1:] = 0

    def moves(name, other):
        """(DC tokens moved, job tokens moved) from obs to other."""
        before, after = _encoder_tokens(name, env, obs), _encoder_tokens(name, env, other)
        return tuple(not np.allclose(a, b, atol=1e-5) for a, b in zip(before, after))

    # A5's encoder reads reach through its cross-attention; V5's and A3's do not.
    assert moves("a5", rerouted) == (True, True)
    assert moves("a5_no_reach", rerouted) == moves("spane", rerouted) == (False, False)
    # V5 still attends across jobs and DCs and A3 does not: a DC token moves when a job leaves.
    assert moves("a5_no_reach", one_less)[0] and not moves("spane", one_less)[0]
    # V3 reads dc_type as one ordinal column instead of the one-hot.
    assert _policy("a5_scalar_dc_type", env).features_extractor.dc_mlp[0].in_features \
        == DC_INPUT_DIM - N_DC_TYPES + 1
    assert _policy("a5", env).features_extractor.dc_mlp[0].in_features == DC_INPUT_DIM


def test_v1_is_a5_under_a_positional_head(make_env):
    # V1 must see what A5 sees, per slot, and differ from it only in the head's symmetry. SB3's
    # head on the pooled context cannot tell which job waits in which slot, nor which DC is in
    # which state.
    from extractors.pointer_policy import TokenHeadPolicy

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    obs = _random_obs(env.observation_space, np.random.default_rng(13), batch=1)
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    obs["jobs_waiting_state"].reshape(n_jobs, 6)[:2] = [[8, 30, 2, 0, 0, 1],    # critical, 8 cores
                                                        [1, 5, 20, 1, 0, 0]]    # tolerant, 1 core
    reach = obs["reach_mask"].reshape(n_jobs, n_slots)
    reach[:2, :N_RING + 2] = 1
    reach[:, 4] = reach[:, 2]                          # edges 2 and 4: the same reach for every job
    swapped = {key: value.copy() for key, value in obs.items()}
    jobs = swapped["jobs_waiting_state"].reshape(n_jobs, 6)
    jobs[[0, 1]] = jobs[[1, 0]]
    traded = {key: value.copy() for key, value in obs.items()}      # edges 2 and 4 trade states
    hosts = traded["infrastructure_state"].reshape(-1, 5)
    two, four = hosts[:, 0] == 2, hosts[:, 0] == 4
    hosts[two, 0], hosts[four, 0] = 4, 2
    v1, a5 = _policy("a5_positional_head", env), _policy("a5", env)
    assert not np.allclose(_logits(v1, swapped)[0, 0], _logits(v1, obs)[0, 0], atol=1e-6)
    # The context stays put when the two DCs trade states, so only the DC tokens can tell V1.
    with torch.no_grad():
        contexts = [v1.features_extractor(_tensors(o)) for o in (obs, traded)]
    torch.testing.assert_close(contexts[1], contexts[0], atol=1e-6, rtol=1e-6)
    assert not np.allclose(_logits(v1, traded)[0, 0], _logits(v1, obs)[0, 0], atol=1e-6)
    assert isinstance(v1, TokenHeadPolicy)
    for part in ("features_extractor", "value_head"):
        shapes = [{key: value.shape for key, value in getattr(policy, part).state_dict().items()}
                  for policy in (v1, a5)]
        assert shapes[0] == shapes[1], part


def _context(name, env, obs):
    with torch.no_grad():
        return _policy(name, env).features_extractor(_tensors(obs)).numpy()


def test_ablations_break_what_they_claim_to_break(make_env):
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(9)
    obs = _random_obs(env.observation_space, rng)
    relabelled, maps = _relabel_with_maps(obs, rng)
    new_id, order = maps[0]

    def relabel_gap(name):
        p, q = _probs(_policy(name, env), obs), _probs(_policy(name, env), relabelled)
        return np.abs(q[0][:, new_id] - p[0][order]).max()

    # V4 (DC-id embedding) loses relabelling equivariance; V2, V3 and V5 keep it.
    assert relabel_gap("a5_dc_id_embedding") > 1e-3
    for name in ("a5_unmasked_pool", "a5_scalar_dc_type", "a5_no_reach"):
        assert relabel_gap(name) < 1e-5, name


def test_v2_unmasks_the_pool_over_dc_slots_only(make_env):
    # V2 reintroduces checklist red flag 3: the DC half of its context averages over every DC
    # slot, so it moves when DCs are added. The job half stays masked.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    half = 64                                   # token_dim: [DC half, job half]
    obs = _random_obs(env.observation_space, np.random.default_rng(3))
    doubled = _with_dcs_doubled(obs)
    v2 = _context("a5_unmasked_pool", env, obs), _context("a5_unmasked_pool", env, doubled)
    assert not np.allclose(v2[0][:, :half], v2[1][:, :half], atol=1e-4)
    np.testing.assert_allclose(_context("a5", env, obs), _context("a5", env, doubled), atol=1e-5)

    # One job against two copies of it, the other job slots padding.
    fewer = {key: value.copy() for key, value in obs.items()}
    n_jobs = SPEC_SHAPE["max_jobs_waiting"]
    jobs = fewer["jobs_waiting_state"].reshape(-1, n_jobs, 6)
    reach = fewer["reach_mask"].reshape(-1, n_jobs, SPEC_SHAPE["max_datacenters"])
    duplicate = np.concatenate([jobs[:, :1]] * 2, axis=1)
    jobs[:] = 0
    jobs[:, :2] = duplicate
    reach[:, 2:] = 0
    reach[:, 2:, 0] = 1
    reach[:, 1] = reach[:, 0]
    single = {key: value.copy() for key, value in fewer.items()}
    single["jobs_waiting_state"].reshape(-1, n_jobs, 6)[:, 1] = 0
    single["reach_mask"].reshape(-1, n_jobs, SPEC_SHAPE["max_datacenters"])[:, 1, 1:] = 0
    np.testing.assert_allclose(_context("a5_unmasked_pool", env, fewer)[:, half:],
                               _context("a5_unmasked_pool", env, single)[:, half:], atol=1e-5)


# ─── TURRET (A4): its output network reads the readout ────────────────────────

def test_turret_actor_is_the_output_network_on_the_readout(make_env):
    # TURRET's output model maps the readout to the actions of all nodes, mu = F_out(S_emb)
    # (Yang et al., AAAI-24), instead of giving each node its own output.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    obs = _random_obs(env.observation_space, np.random.default_rng(7))
    policy = _policy("turret", env)
    before = _logits(policy, obs)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in policy.features_extractor.readout.parameters():
            p.add_(torch.randn_like(p))
    assert not np.allclose(_logits(policy, obs), before, atol=1e-4)


def test_turret_actor_is_positional_in_jobs_and_dcs(make_env):
    # The readout is invariant to job order and DC numbering, so F_out keeps each job slot's
    # logits when the jobs are permuted, and each action's logit when the DCs are renumbered.
    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    rng = np.random.default_rng(7)
    obs = _random_obs(env.observation_space, rng)
    policy = _policy("turret", env)
    n_jobs = SPEC_SHAPE["max_jobs_waiting"]
    order = rng.permutation(n_jobs)
    jobs_only = {key: value.copy() for key, value in obs.items()}
    for key, width in (("jobs_waiting_state", 6), ("reach_mask", SPEC_SHAPE["max_datacenters"])):
        jobs_only[key] = obs[key].reshape(-1, n_jobs, width)[:, order].reshape(obs[key].shape)
    np.testing.assert_allclose(_logits(policy, jobs_only), _logits(policy, obs), atol=1e-5)

    relabelled, maps = _relabel_with_maps(obs, rng)
    p, q = _logits(policy, obs), _logits(policy, relabelled)
    np.testing.assert_allclose(q, p, atol=1e-5)
    new_id, order = maps[0]
    assert not np.allclose(q[0][:, new_id], p[0][order], atol=1e-3)


@pytest.mark.parametrize("name", ["spane", "a5_positional_head"])
def test_token_head_policies_survive_save_and_load(name, make_env, tmp_path):
    from sb3_contrib import MaskablePPO

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    model = _model(name, env)
    model.save(tmp_path / "m")
    loaded = MaskablePPO.load(tmp_path / "m", device="cpu")
    assert type(loaded.policy) is type(model.policy)
    obs = _random_obs(env.observation_space, np.random.default_rng(6), batch=2)
    np.testing.assert_allclose(_probs(model.policy.eval(), obs), _probs(loaded.policy.eval(), obs))


# ─── Fine-tuning scopes ───────────────────────────────────────────────────────

def _finetune_group(name: str) -> str:
    if name.startswith(("features_extractor.", "pi_features_extractor.", "vf_features_extractor.")):
        return "extractor"
    if name.startswith(("value_head.", "value_net.", "mlp_extractor.value_net.")):
        return "critic"
    return "actor head"


SCOPES = {"full": {"extractor", "critic", "actor head"}, "head": {"actor head", "critic"},
          "extractor": {"extractor", "critic"}}


def _adam_step(policy):
    policy.optimizer.zero_grad()
    sum(p.sum() for p in policy.parameters() if p.requires_grad).backward()
    policy.optimizer.step()


@pytest.mark.parametrize("name", ["euromlsys", "deepsets", "spane", "turret", "a5",
                                  "a5_positional_head"])
def test_finetune_scopes_train_the_critic_and_start_a_fresh_optimizer(name, make_env, tmp_path):
    # Each scoped arm is full fine-tuning minus one part of the actor, and every arm, full
    # included, starts from fresh Adam moments, as training from scratch does.
    from sb3_contrib import MaskablePPO
    from utils.misc import apply_finetune_scope

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    source = _model(name, env)
    _adam_step(source.policy)                       # the source run's optimizer has moments
    source.save(tmp_path / "source")
    groups = {key: _finetune_group(key) for key, _ in source.policy.named_parameters()}
    assert set(groups.values()) == SCOPES["full"]
    for scope, trained in SCOPES.items():
        model = MaskablePPO.load(tmp_path / "source", device="cpu")
        assert model.policy.optimizer.state         # SB3's load restores them
        apply_finetune_scope(model, scope)
        policy = model.policy
        expected = {key for key, group in groups.items() if group in trained}
        assert {key for key, p in policy.named_parameters() if p.requires_grad} == expected
        optimizer = policy.optimizer
        assert not optimizer.state
        assert [p for group in optimizer.param_groups for p in group["params"]] \
            == list(policy.parameters())
        before = {key: p.detach().clone() for key, p in policy.named_parameters()}
        _adam_step(policy)
        assert {key for key, p in policy.named_parameters()
                if not torch.equal(p, before[key])} == expected, scope
        model.save(tmp_path / scope)
        MaskablePPO.load(tmp_path / scope, device="cpu")    # what evaluate loads
    with pytest.raises(ValueError):
        apply_finetune_scope(model, "encoder")


# ─── Deterministic evaluation (utils.evaluation.model_predictor) ─────────────

def _idle_observation(env, jobs):
    """An idle cluster with `jobs` = [(origin DC name, cores, runtime, time_to_due,
    sensitivity)] waiting, through the env's own observation path."""
    datacenters = env.params["datacenters"]
    names = [dc["name"] for dc in datacenters]
    hosts = []
    for dc_id, dc in enumerate(datacenters, start=1):
        for host in dc["hosts"]:
            hosts += [[dc_id, env.DC_TYPE_IDS[dc["type"]], host["pes"], host["pes"], 0]] * host["amount"]
    wire = []
    for origin, cores, runtime, due, sensitivity in jobs:
        wire += [cores, names.index(origin), runtime, due] + [int(s == sensitivity) for s in range(3)]
    return env._get_observation({"infrastructure_observation": np.ravel(hosts).tolist(),
                                 "secondary_observation": wire})


def _two_idle_edges(name, env):
    """A pointer policy that places jobs, the DC names by action, and an idle cluster where a
    micro_dc_04 job may go only to edge_dc_03 or edge_dc_05 (or wait): the two edges are in
    identical states."""
    names = ["no-op"] + [dc["name"] for dc in env.params["datacenters"]]
    obs = _idle_observation(env, [("micro_dc_04", 2, 10, 20, 1)])
    mask = np.zeros((SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]), dtype=bool)
    mask[:, 0] = True
    mask[0, [names.index("edge_dc_03"), names.index("edge_dc_05")]] = True
    policy = _policy(name, env)
    with torch.no_grad():
        policy.noop_head[-1].bias.fill_(-5.0)                         # place the job
    return policy, names, obs, mask.ravel()


@pytest.mark.parametrize("name", ["spane", "a5"])
def test_deterministic_evaluation_breaks_exact_ties_by_dc_name(name, make_env):
    # Two idle edges that a micro_dc_04 job may use score exactly alike under a pointer head.
    # SB3's argmax takes the lower action index, edge_dc_03 on S but edge_dc_05 on PI-S, where
    # the same DCs are numbered differently; the tie rule picks edge_dc_03 on both.
    from types import SimpleNamespace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.evaluation import model_predictor

    chosen = {}
    for member in ("S", "PI-S"):
        env = make_env(datacenters=ring_member(member)[0], split_large_jobs=False, **SPEC_SHAPE)
        policy, names, obs, mask = _two_idle_edges(name, env)
        batch, masks = {key: value[None] for key, value in obs.items()}, mask[None]
        sb3, _ = policy.predict(batch, action_masks=masks, deterministic=True)
        predict = model_predictor(SimpleNamespace(policy=policy), DummyVecEnv([lambda: env]))
        ours = predict(batch, masks)
        chosen[member] = names[sb3[0, 0]], names[ours[0, 0]], int(predict.tie_breaks[0])
    assert (chosen["S"][0], chosen["PI-S"][0]) == ("edge_dc_03", "edge_dc_05")
    assert chosen["S"][1:] == chosen["PI-S"][1:] == ("edge_dc_03", 1)


@pytest.mark.parametrize("name", ["spane", "a5"])
def test_deterministic_evaluation_sees_the_tie_on_every_row_of_a_batch(name, make_env):
    # evaluate steps its 16 workers with one forward pass. With 16 or more CPU threads the
    # kernels round some batch rows differently, and the two idle edges' logits came out a few
    # ulps apart on 4-6 of the 16 rows, where the tie rule then never saw the tie.
    from types import SimpleNamespace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.evaluation import model_predictor

    env = make_env(datacenters=ring_member("S")[0], split_large_jobs=False, **SPEC_SHAPE)
    policy, names, obs, mask = _two_idle_edges(name, env)
    predict = model_predictor(SimpleNamespace(policy=policy), DummyVecEnv([lambda: env]))
    others = _random_obs(env.observation_space, np.random.default_rng(0), batch=16)
    threads = torch.get_num_threads()
    torch.set_num_threads(20)
    try:
        for row in range(16):
            batch = {key: value.copy() for key, value in others.items()}
            masks = batch["reach_mask"].astype(bool)
            for key in batch:
                batch[key][row] = obs[key]
            masks[row] = mask
            actions = predict(batch, masks)
            assert (names[actions[row, 0]], predict.tie_breaks[row]) == ("edge_dc_03", 1), row
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("name", ["euromlsys", "a5"])
def test_deterministic_evaluation_keeps_every_strictly_best_action(name, make_env):
    from types import SimpleNamespace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.evaluation import model_predictor

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    obs = _random_obs(env.observation_space, np.random.default_rng(11))
    masks = obs["reach_mask"].astype(bool)
    policy = _policy(name, env)
    predict = model_predictor(SimpleNamespace(policy=policy), DummyVecEnv([lambda: env]))
    sb3, _ = policy.predict(obs, action_masks=masks, deterministic=True)
    np.testing.assert_array_equal(predict(obs, masks), sb3)
    assert (predict.tie_breaks == 0).all()


def test_deterministic_evaluation_settles_exact_ties_only(make_env):
    # Logits set through SB3's action_net (zero weights, so the logits are its bias): a DC one
    # float32 ulp ahead of the DC whose name sorts first keeps its lead; exactly level, the
    # name decides.
    from types import SimpleNamespace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.evaluation import model_predictor

    env = make_env(datacenters=ring_topology(N_RING), **SPEC_SHAPE)
    names = ["no-op"] + [dc["name"] for dc in env.params["datacenters"]]
    first, leader = names.index(min(names[1:])), names.index("micro_2")
    n_jobs, n_slots = SPEC_SHAPE["max_jobs_waiting"], SPEC_SHAPE["max_datacenters"]
    masks = np.zeros((1, n_jobs, n_slots), dtype=bool)
    masks[..., 0] = True
    masks[0, 0, [first, leader]] = True
    obs = _random_obs(env.observation_space, np.random.default_rng(0), batch=1)
    policy = _policy("euromlsys", env)
    predict = model_predictor(SimpleNamespace(policy=policy), DummyVecEnv([lambda: env]))
    one = np.float32(1.0)
    for runner_up, chosen, ties in ((np.nextafter(one, np.float32(0.0)), leader, 0),
                                    (one, first, 1)):
        logits = np.full((n_jobs, n_slots), -1.0, dtype=np.float32)
        logits[0, [leader, first]] = one, runner_up
        with torch.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.copy_(torch.from_numpy(logits.ravel()))
        actions = predict(obs, masks.reshape(1, -1))
        assert (actions[0, 0], predict.tie_breaks[0]) == (chosen, ties)

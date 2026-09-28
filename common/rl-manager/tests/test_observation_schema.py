"""Observation schema and action masking of JobPlacementEnv, checked without a Java gateway."""

import os
import re

import numpy as np
import pytest
import yaml

from conftest import REPO, SPEC_SHAPE, ring_topology

JAVA_JP = os.path.join(REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "main",
                       "java", "daislab", "cspg")


def _java_int(file_name: str, constant: str) -> int:
    source = open(os.path.join(JAVA_JP, file_name)).read()
    return int(re.search(rf"static final int {constant} = (\d+);", source).group(1))


def test_java_and_python_agree_on_the_wire_layout():
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    assert _java_int("WrappedSimulation.java", "HOST_OBS_FEATURES") == JobPlacementEnv.HOST_OBS_FEATURES
    assert _java_int("CloudSimProxy.java", "JOB_OBS_FEATURES") == JobPlacementEnv._JOB_WIRE_FEATURES
    assert JobPlacementEnv._JOB_WIRE_FEATURES == JobPlacementEnv.JOB_OBS_FEATURES + 1  # + location
    java_types = dict(re.findall(
        r'case "(\w+)" -> (\d+);', open(os.path.join(JAVA_JP, "WrappedSimulation.java")).read()))
    assert {k: int(v) for k, v in java_types.items()} == JobPlacementEnv.DC_TYPE_IDS


def test_config_pins_the_benchmark_shape_constants():
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    config = yaml.load(open(os.path.join(REPO, "domain", "job-placement", "config.yml")), Loader)
    assert {k: config["common"][k] for k in SPEC_SHAPE} == SPEC_SHAPE


def test_benchmark_observation_is_960_plus_192_plus_768(make_env):
    spaces = make_env(datacenters=ring_topology(18), **SPEC_SHAPE).observation_space.spaces
    shapes = {key: space.shape[0] for key, space in spaces.items()}
    assert shapes == {"infrastructure_state": 960, "jobs_waiting_state": 192, "reach_mask": 768}


def _raw_obs(env, jobs, free_by_dc=None):
    """Java-format observation for env's topology: hosts in DC order, jobs as wire rows."""
    types = env.DC_TYPE_IDS
    hosts = []
    for idx, dc in enumerate(env.params["datacenters"]):
        for host in dc["hosts"]:
            for _ in range(host["amount"]):
                free = host["pes"] if free_by_dc is None else free_by_dc[idx]
                hosts.append([idx + 1, types[dc["type"]], host["pes"], free, 0])
    return {"infrastructure_observation": np.ravel(hosts).tolist(),
            "secondary_observation": np.ravel(jobs).tolist()}


def _job(cores, location, sensitivity=0):
    onehot = [int(s == sensitivity) for s in range(3)]
    return [cores, location, 5, 7] + onehot


def _legal_actions(location, n_ring):
    """No-op, the origin, its ring neighbours and the cloud (action = DC index + 1)."""
    pos = location - 1
    return {0, location + 1, (pos - 1) % n_ring + 2, (pos + 1) % n_ring + 2, 1}


def test_reach_mask_follows_the_topology_and_padding_takes_only_the_noop(make_env):
    n_ring = 6
    env = make_env(datacenters=ring_topology(n_ring), **SPEC_SHAPE)
    jobs = [_job(2, 1), _job(8, 2, 2), _job(4, 6, 1)]
    obs = env._get_observation(_raw_obs(env, jobs))

    reach = obs["reach_mask"].reshape(env.max_jobs_waiting, env.max_datacenters)
    for j, job in enumerate(jobs):
        assert set(np.flatnonzero(reach[j])) == _legal_actions(job[1], n_ring)
    assert (reach[len(jobs):, 0] == 1).all() and not reach[len(jobs):, 1:].any()

    # Location is stripped from the policy's job features.
    policy_jobs = obs["jobs_waiting_state"].reshape(env.max_jobs_waiting, env.JOB_OBS_FEATURES)
    assert policy_jobs[1].tolist() == [8, 5, 7, 0, 0, 1]


def _shrink_vms(topology, dc_indices, pes):
    for idx in dc_indices:
        for host in topology[idx]["hosts"]:
            host["pes"] = pes
            for vm in host["vms"]:
                vm["pes"] = pes
    return topology


def test_a_full_dc_stays_legal_because_the_job_queues_there(make_env):
    env = make_env(datacenters=ring_topology(6), **SPEC_SHAPE)
    jobs = [_job(2, 1), _job(8, 2), _job(4, 6)]
    no_free_pes = [0] * len(env.params["datacenters"])
    obs = env._get_observation(_raw_obs(env, jobs, free_by_dc=no_free_pes))
    mask = np.array(env.action_masks()).reshape(env.max_jobs_waiting, env.max_datacenters)
    assert (mask == obs["reach_mask"].reshape(mask.shape).astype(bool)).all()


def test_action_mask_is_reach_restricted_by_vm_capacity(make_env):
    n_ring = 6
    # DC 2 (micro) gets 4-PE VMs: an 8-core job can never run there.
    env = make_env(datacenters=_shrink_vms(ring_topology(n_ring), [2], 4), **SPEC_SHAPE)
    jobs = [_job(2, 1), _job(8, 2), _job(4, 6)]
    obs = env._get_observation(_raw_obs(env, jobs))
    mask = np.array(env.action_masks()).reshape(env.max_jobs_waiting, env.max_datacenters)
    reach = obs["reach_mask"].reshape(mask.shape).astype(bool)
    assert not (mask & ~reach).any()
    assert set(np.flatnonzero(mask[1])) == _legal_actions(2, n_ring) - {3}
    assert set(np.flatnonzero(mask[0])) == _legal_actions(1, n_ring)
    assert mask[:, 0].all()
    assert not mask[len(jobs):, 1:].any()


def test_job_that_fits_no_reachable_dc_can_only_wait(make_env):
    # Used to fall back to "every action valid", which let the agent place the job on a DC
    # the topology forbids. The no-op already keeps MaskablePPO's sub-space non-empty.
    env = make_env(datacenters=_shrink_vms(ring_topology(6), [0, 3, 4, 5], 4), **SPEC_SHAPE)
    env._get_observation(_raw_obs(env, [_job(8, 4)]))
    mask = np.array(env.action_masks()).reshape(env.max_jobs_waiting, env.max_datacenters)
    assert set(np.flatnonzero(mask[0])) == {0}


def test_pad_observation_refuses_to_truncate(make_env):
    env = make_env()
    assert env._pad_observation(np.arange(3, dtype=np.int32), 5).tolist() == [0, 1, 2, 0, 0]
    with pytest.raises(ValueError, match="holds 5"):
        env._pad_observation(np.arange(6, dtype=np.int32), 5)


def test_topology_with_more_hosts_than_total_hosts_is_rejected(make_env):
    with pytest.raises(ValueError, match="total_hosts=10"):
        make_env(total_hosts=10)  # Env B has 29 hosts

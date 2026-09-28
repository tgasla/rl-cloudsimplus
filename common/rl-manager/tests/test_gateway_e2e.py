"""The observation's column layout, checked against the live job-placement gateway.

Skipped unless the gateway jar has been built (make build-gateway domain=job-placement).
"""

import json
import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from conftest import REPO, SPEC_SHAPE, ring_topology

JAR = os.path.join(REPO, "domain", "job-placement", "cloudsimplus-gateway", "build", "libs",
                   "cloudsimplus-gateway-0.1.0.jar")
pytestmark = pytest.mark.skipif(not (os.path.exists(JAR) and shutil.which("java")),
                                reason="job-placement gateway jar not built")
N_RING = 4


@pytest.fixture(scope="module")
def gateway_port():
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    java = subprocess.Popen(
        ["java", "-Dlog.level=WARN", "-Dlog.destination=none", "-Dlog.saveExperiment=false",
         "-jar", JAR, "--grpc", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("localhost", port)) == 0:
                break
        time.sleep(0.1)
    yield port
    java.terminate()
    java.wait(timeout=10)


@pytest.fixture
def live_env(gateway_port):
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    params = json.load(open(os.path.join(
        REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "test", "resources",
        "env_b_params.json")))
    params.update(SPEC_SHAPE, datacenters=ring_topology(N_RING), split_large_jobs=False)
    # Job j: 2 cores, 10 s at 60 MIPS, arriving at t=1 at ring DC j+1.
    jobs = [{"jobId": j, "submissionDelay": 1, "mi": 600, "cores": 2, "location": j + 1,
             "delaySensitivity": j % 3, "deadline": 40} for j in range(N_RING)]
    env = JobPlacementEnv(params, jobs_as_json=json.dumps(jobs), port=gateway_port)
    yield env
    env.close()


def test_host_columns_match_the_topology(live_env):
    obs, _ = live_env.reset()
    hosts = obs["infrastructure_state"].reshape(-1, 5)
    real = hosts[hosts[:, 0] > 0]
    topology = live_env.params["datacenters"]
    assert len(real) == sum(h["amount"] for dc in topology for h in dc["hosts"])
    for dc_id, dc_type, capacity, free, backlog in real:
        dc = topology[dc_id - 1]
        assert dc_type == live_env.DC_TYPE_IDS[dc["type"]]
        assert capacity == dc["hosts"][0]["pes"] == free
        assert backlog == 0


def test_job_location_column_drives_reach_and_placement_uses_free_pes(live_env):
    live_env.reset()
    obs, *_ = live_env.step(np.zeros(live_env.max_jobs_waiting, dtype=int))
    reach = obs["reach_mask"].reshape(live_env.max_jobs_waiting, live_env.max_datacenters)
    jobs = obs["jobs_waiting_state"].reshape(live_env.max_jobs_waiting, 6)
    assert (jobs[:N_RING, 0] == 2).all() and (jobs[:N_RING, 3:].sum(1) == 1).all()
    for j in range(N_RING):
        origin = j + 1
        neighbours = {(origin - 2) % N_RING + 1, origin % N_RING + 1}
        expected = {0, 1, origin + 1} | {n + 1 for n in neighbours}
        assert set(np.flatnonzero(reach[j])) == expected

    # Place slot 0 on its own DC (micro or edge, no or short network delay).
    action = np.zeros(live_env.max_jobs_waiting, dtype=int)
    origin_action = int(np.flatnonzero(reach[0])[2])
    action[0] = origin_action
    before = obs["infrastructure_state"].reshape(-1, 5)
    for _ in range(2):  # allow the edge's 1 s network delay
        obs, *_ = live_env.step(action)
        action[:] = 0
    after = obs["infrastructure_state"].reshape(-1, 5)
    in_dc = after[:, 0] == origin_action
    assert before[in_dc, 3].sum() - after[in_dc, 3].sum() == 2
    assert after[in_dc, 4].sum() > 0

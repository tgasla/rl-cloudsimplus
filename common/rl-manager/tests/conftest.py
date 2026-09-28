"""Shared helpers for the rl-manager tests. Run: python3 -m pytest common/rl-manager/tests"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                  # rl-manager
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gym_cloudsimplus"))

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ENV_B_PARAMS = os.path.join(
    REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "test", "resources",
    "env_b_params.json",
)

# The benchmark's shape constants (docs/analysis/05-benchmark-redesign-ring-n.json).
SPEC_SHAPE = {
    "max_datacenters": 24,
    "max_hosts": 8,
    "total_hosts": 192,
    "max_host_pes": 32,
    "max_jobs_waiting": 32,
    "max_job_pes": 8,
}

TIER_HOSTS = {"cloud": (2, 32), "edge": (2, 16), "micro": (1, 8)}  # (hosts, PEs per host)


def ring_topology(n_ring: int) -> list:
    """Cloud at index 0, then n_ring DCs alternating edge/micro; each ring DC connects to
    its two ring neighbours and the cloud (connect_to holds DC indices, as after entrypoint)."""
    dcs = []
    for idx, dc_type in enumerate(["cloud"] + ["edge", "micro"] * n_ring):
        if idx > n_ring:
            break
        amount, pes = TIER_HOSTS[dc_type]
        # VM size must exceed a cloudlet's file size (one MTU) or no cloudlet fits.
        vm = {"amount": 1, "pes": pes, "pe_mips": 60, "ram": 65536, "size": 1000000,
              "bw": 10000}
        host = {"amount": amount, "pes": pes, "pe_mips": 60, "ram": 65536, "storage": 1000000,
                "bw": 10000, "vms": [vm]}
        ring_pos = idx - 1
        connect_to = [] if idx == 0 else [
            (ring_pos - 1) % n_ring + 1, (ring_pos + 1) % n_ring + 1, 0,
        ]
        dcs.append({"name": f"{dc_type}_{idx}", "type": dc_type, "amount": 1,
                    "hosts": [host], "connect_to": connect_to})
    return dcs


@pytest.fixture
def make_env(monkeypatch):
    """Build a JobPlacementEnv from the Env B fixture params without a Java gateway."""
    from gym_cloudsimplus.cloud_sim_grpc_client import CloudSimGrpcClient
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    monkeypatch.setattr(CloudSimGrpcClient, "create_simulation", lambda self, *a, **k: "sim")
    created = []

    def make(**overrides):
        params = json.load(open(ENV_B_PARAMS))
        params.update(overrides)
        env = JobPlacementEnv(params)
        created.append(env)
        return env

    yield make
    for env in created:
        env._client.close_channel()

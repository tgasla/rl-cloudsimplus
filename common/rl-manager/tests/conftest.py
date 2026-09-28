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

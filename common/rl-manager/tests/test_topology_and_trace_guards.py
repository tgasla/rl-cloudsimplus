"""Guards on the job-placement input path. Run: python3 -m pytest common/rl-manager/tests"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                  # rl-manager
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gym_cloudsimplus"))

from utils.misc import _check_datacenter_amounts_are_one  # noqa: E402
from utils.trace_utils import csv_to_cloudlet_descriptor  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ENV_B_PARAMS = os.path.join(
    REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "test", "resources",
    "env_b_params.json",
)


def test_csv_deadline_reaches_the_job_descriptor(tmp_path):
    # The deadline column used to be read and dropped, so Java saw 0 for every job.
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "job_id,arrival_time,mi,required_cores,location,delay_sensitivity,deadline\n"
        "0,1,600,2,micro_dc_ucd,tolerant,3\n"
        "1,2,1200,4,micro_dc_dcu,critical,5\n"
    )
    jobs = csv_to_cloudlet_descriptor(str(trace))
    assert [job["deadline"] for job in jobs] == [3, 5]
    assert all(isinstance(job["deadline"], int) for job in jobs)


def test_topology_larger_than_the_action_space_is_rejected():
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    params = json.load(open(ENV_B_PARAMS))
    params["max_datacenters"] = 8                       # addresses 7 real DCs; action 0 is no-op
    template = params["datacenters"][0]
    params["datacenters"] = [dict(template, name=f"dc{i}", connect_to=[]) for i in range(8)]
    with pytest.raises(ValueError, match="addresses only 7"):
        JobPlacementEnv(params)


def test_topology_that_fills_the_action_space_is_accepted(make_env):
    template = json.load(open(ENV_B_PARAMS))["datacenters"][0]
    dcs = [dict(template, name=f"dc{i}", connect_to=[]) for i in range(7)]
    assert make_env(max_datacenters=8, datacenters=dcs).max_datacenters == 8


def test_datacenter_amount_other_than_one_is_rejected():
    _check_datacenter_amounts_are_one([{"name": "a", "amount": 1}, {"name": "b"}])
    with pytest.raises(ValueError, match="'c'"):
        _check_datacenter_amounts_are_one([{"name": "a", "amount": 1}, {"name": "c", "amount": 2}])


def test_reset_forwards_jobs_json_into_the_grpc_request(make_env):
    from gym_cloudsimplus.protos.unified import cloudsimplus_pb2 as pb2

    env = make_env()
    sent = []

    class FakeStub:
        def reset(self, request):
            sent.append(request)
            return pb2.ResetResult()

    env._client.stub = FakeStub()
    env.reset(options={"jobs_json": '[{"jobId": 7}]'})
    env.reset()
    assert [request.jobs_json for request in sent] == ['[{"jobId": 7}]', ""]


def test_ring_runs_refuse_a_horizon_that_ends_before_the_arrivals():
    from utils.misc import level_stream

    base = {"max_episode_length": 150, "timestep_interval": 1.0, "ring_manifest": "unused",
            "benchmark_member": "S", "datacenters": [], "level_split": "train", "seed": 0}
    with pytest.raises(ValueError, match="max_episode_length >= 200"):
        level_stream(base, 0)
    with pytest.raises(ValueError, match="timestep_interval 1.0"):
        level_stream(dict(base, max_episode_length=200, timestep_interval=2.0), 0)


def test_test_mode_refuses_ring_runs():
    from test import test as run_test

    with pytest.raises(ValueError, match="mode: evaluate"):
        run_test({"benchmark_member": "S"}, [])


def test_topologies_the_action_mask_cannot_describe_are_rejected(make_env):
    template = json.load(open(ENV_B_PARAMS))["datacenters"]
    two_vms = json.loads(json.dumps(template))
    vm = two_vms[1]["hosts"][0]["vms"][0]
    two_vms[1]["hosts"][0]["vms"] = [dict(vm, pes=8), dict(vm, pes=8)]
    with pytest.raises(ValueError, match="exactly one VM"):
        make_env(datacenters=two_vms)
    tiny = json.loads(json.dumps(template))
    tiny[2]["hosts"][0]["vms"][0]["size"] = 1000
    with pytest.raises(ValueError, match="cannot hold a cloudlet"):
        make_env(datacenters=tiny)

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


def test_datacenter_amount_other_than_one_is_rejected():
    _check_datacenter_amounts_are_one([{"name": "a", "amount": 1}, {"name": "b"}])
    with pytest.raises(ValueError, match="'c'"):
        _check_datacenter_amounts_are_one([{"name": "a", "amount": 1}, {"name": "c", "amount": 2}])

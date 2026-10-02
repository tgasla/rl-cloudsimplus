"""The differential harness benchmark/tools/diff_port.py against the live gateway jar.

Run: python3 -m pytest benchmark/tests/test_diff_port.py (skipped when the jar is not built)
"""
import os
import shutil
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import cloudsim_port as cp  # noqa: E402
import diff_port as dp  # noqa: E402

pytestmark = pytest.mark.skipif(not (os.path.exists(cp.JAR_PATH) and shutil.which("java")),
                                reason="job-placement gateway jar not built")


@pytest.mark.parametrize("name, shut_down", [("shutdown", True), ("alone", False)])
def test_a_jar_that_shut_down_once_every_job_finished_is_no_difference(name, shut_down):
    # Both scenarios finish every job inside the last step. In "shutdown" CloudSim then runs out
    # of events before the step's target and the jar's last observation lists no hosts; in
    # "alone" a keep-alive event is still pending, so the jar lists them and they are compared.
    report = dp.play_scenario(name)
    assert report["problems"] == []
    assert report["java_shut_down"] is shut_down


def test_each_gateway_writes_its_logback_config_to_a_directory_of_its_own(tmp_path, monkeypatch):
    # A gateway that logs writes its logback config to <log.simDir>/logback-generated.xml, by
    # default <working directory>/logs, and parses it back. Gateways started together from one
    # working directory parsed it while another had just truncated it, and exited.
    monkeypatch.chdir(tmp_path)
    gateways = []
    try:
        for i in range(2):
            gateways.append(dp.Gateway(log_path=str(tmp_path / f"gateway{i}.log")))
        assert not (tmp_path / "logs").exists()
        configs = {os.path.join(g.log_dir, "logback-generated.xml") for g in gateways}
        assert len(configs) == 2 and all(os.path.exists(c) for c in configs)
    finally:
        for g in gateways:
            g.close()
    assert not any(os.path.exists(g.log_dir) for g in gateways)


def test_a_gateway_that_fails_to_start_leaves_no_directory_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="exited"):
        dp.Gateway(jar=str(tmp_path / "missing.jar"), log_path=str(tmp_path / "gateway.log"))
    assert os.listdir(tmp_path) == ["gateway.log"]

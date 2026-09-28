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


def _ring_member(member_id):
    from utils import levels
    from utils.misc import _translate_connect_to_names_to_idx

    ring = os.path.join(REPO, "common", "topologies", "ring")
    topology = levels.load_topology(os.path.join(ring, f"{member_id}.yml"))
    for dc in topology:
        dc["connect_to"] = levels._as_list(dc.get("connect_to", []))
        dc["hosts"] = levels._as_list(dc["hosts"])
        for host in dc["hosts"]:
            host["vms"] = levels._as_list(host["vms"])
    return _translate_connect_to_names_to_idx(topology), os.path.join(ring, "manifest.json")


@pytest.fixture
def level_env(gateway_port):
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv
    from utils.misc import level_stream

    def make(member_id, split="train", rank=0):
        datacenters, manifest = _ring_member(member_id)
        params = json.load(open(os.path.join(
            REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "test", "resources",
            "env_b_params.json")))
        params.update(SPEC_SHAPE, datacenters=datacenters, split_large_jobs=False, seed=3,
                      max_episode_length=200, benchmark_member=member_id,
                      ring_manifest=manifest, level_split=split, num_cpu=16)
        env = JobPlacementEnv(params, jobs_as_json="[]", port=gateway_port)
        env.set_level_stream(*level_stream(params, rank))
        made.append(env)
        return env

    made = []
    yield make
    for env in made:
        env.close()


def _episode(env, policy):
    """Unshaped return and level id of one episode from a fresh reset."""
    obs, info = env.reset()
    level, total, done = info["level_id"], 0.0, False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(env))
        total += info["unshaped_reward"]
        done = terminated or truncated
    return level, total


def _all_to_cloud(env):
    mask = np.array(env.action_masks()).reshape(env.max_jobs_waiting, env.max_datacenters)
    return np.where(mask[:, 1], 1, 0)


def test_each_reset_plays_a_new_level_and_a_level_replays_exactly(level_env):
    env = level_env("S")
    first, first_return = _episode(env, _all_to_cloud)
    second, second_return = _episode(env, _all_to_cloud)
    assert first != second and first_return != second_return

    replay = level_env("S")
    replay._level_sampler.next = lambda: first
    assert _episode(replay, _all_to_cloud) == (first, first_return)


def test_sb3_auto_reset_moves_on_to_the_next_level(level_env):
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = level_env("S")
    vec = DummyVecEnv([lambda: env])
    vec.reset()
    start = vec.reset_infos[0]["level_id"]
    done = [False]
    while not done[0]:
        _, _, done, infos = vec.step(np.array([np.zeros(env.max_jobs_waiting, dtype=int)]))
    assert infos[0]["level_id"] == start          # the finished episode's level
    assert vec.reset_infos[0]["level_id"] != start


def test_largest_member_resets_fast_enough(level_env):
    env = level_env("C1-N19", split="test")
    payload = env._levels.jobs_json(env._level_sampler.next())
    env.reset()                                    # warm the JVM and the cache
    start = time.perf_counter()
    env.reset(options={"jobs_json": payload})
    elapsed = time.perf_counter() - start
    n_jobs = payload.count('"jobId"')
    print(f"C1-N19: {n_jobs} jobs, {len(payload) / 1024:.0f} KB, reset {1000 * elapsed:.1f} ms")
    assert n_jobs > 2500 and elapsed < 0.25


def test_parallel_vec_env_matches_sequential_dummy_vec_env(level_env):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.misc import ParallelBatchDummyVecEnv

    def rollout(vec_cls, steps=260):  # long enough to cross an auto-reset
        envs = [level_env("S", rank=rank) for rank in range(4)]
        vec = vec_cls([lambda env=env: env for env in envs])
        obs = vec.reset()
        trace, start = [], time.perf_counter()
        for _ in range(steps):
            actions = np.stack([_all_to_cloud(env) for env in envs])
            obs, rewards, dones, infos = vec.step(actions)
            trace.append((obs["jobs_waiting_state"].copy(), rewards.copy(), dones.copy(),
                          [info["level_id"] for info in infos]))
        return trace, time.perf_counter() - start

    sequential, t_seq = rollout(DummyVecEnv)
    parallel, t_par = rollout(ParallelBatchDummyVecEnv)
    print(f"4 workers, 260 steps: sequential {t_seq:.2f} s, parallel {t_par:.2f} s")
    assert any(dones.any() for _, _, dones, _ in sequential)
    for (o1, r1, d1, l1), (o2, r2, d2, l2) in zip(sequential, parallel):
        assert (o1 == o2).all() and (r1 == r2).all() and (d1 == d2).all() and l1 == l2


def _free_port_block(n):
    """The first port of n consecutive free ports."""
    for base in range(52000, 60000, n):
        socks = []
        try:
            for port in range(base, base + n):
                s = socket.socket()
                s.bind(("", port))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


@pytest.fixture
def spawned_params(monkeypatch, tmp_path):
    """Params for runs that spawn their own JVMs through utils.misc (as in the container)."""
    monkeypatch.setenv("CLOUDSIM_GATEWAY_JAR", JAR)
    monkeypatch.setenv("JAVA_LOG_DESTINATION", "none")
    datacenters, manifest = _ring_member("S")
    params = json.load(open(os.path.join(
        REPO, "domain", "job-placement", "cloudsimplus-gateway", "src", "test", "resources",
        "env_b_params.json")))
    params.update(SPEC_SHAPE, datacenters=datacenters, split_large_jobs=False, seed=3,
                  max_episode_length=200, benchmark_member="S", ring_manifest=manifest,
                  log_dir=str(tmp_path), save_experiment=True, grpc_base_port=_free_port_block(16))
    return params


def test_evaluate_plays_each_val_level_once(spawned_params, tmp_path):
    from evaluate import evaluate
    from utils import levels

    spawned_params.update(level_split="val", num_cpu=4,
                          cloudlet_to_dc_mapping="earliest-shortest-to-most-free-dc")
    df = evaluate(spawned_params, [])
    assert sorted(df["level_id"]) == list(levels.VAL_LEVELS)
    assert (tmp_path / "evaluation.csv").exists()
    assert df["unshaped_return"].std() > 0           # contexts differ
    assert df["terminated"].all()


def test_training_keeps_best_val_and_final_models(spawned_params, tmp_path):
    from sb3_contrib import MaskablePPO
    from utils.misc import create_val_callback, vectorize_env

    spawned_params.update(level_split="train", num_cpu=1, val_num_cpu=2, val_every=64,
                          cloudlet_to_dc_mapping="rl")
    env = vectorize_env(None, MaskablePPO, num_cpu=1, params=spawned_params, jobs_json="[]")
    callback, val_env = create_val_callback(spawned_params, num_cpu=1)
    try:
        model = MaskablePPO("MultiInputPolicy", env, n_steps=64, batch_size=64, n_epochs=1,
                            seed=0, device="cpu")
        model.learn(128, callback=callback)
    finally:
        env.close()
        val_env.close()
    import pandas as pd
    from utils import levels

    val = pd.read_csv(tmp_path / "val.csv")
    assert sorted(val["timestep"].unique()) == [64, 128]
    for _, sweep in val.groupby("timestep"):
        assert sorted(sweep["level_id"]) == list(levels.VAL_LEVELS)
    assert (tmp_path / "best_val_model.zip").exists() and (tmp_path / "final_model.zip").exists()
    best = val.groupby("timestep")["unshaped_return"].mean().max()
    assert callback.best_mean == pytest.approx(best)

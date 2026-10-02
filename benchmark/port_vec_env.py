"""Vectorised RING-N training envs backed by the Python port -- no JVM, no gRPC, no Docker.

`utils.misc.vectorize_env` builds `ParallelBatchDummyVecEnv`: a DummyVecEnv that steps its workers
on a thread pool. That is the right shape for the gRPC path, where each `step` blocks on a socket
and releases the GIL, so threads genuinely overlap the JVMs. The port is in-process Python, so the
same trick buys nothing -- the GIL serialises it. Parallelism here needs processes.

Measured on member S, 200-step level, one process (benchmark/port_client.py):

    env.step through JobPlacementEnv + PortClient    4,780 steps/s   0.21 ms/step
    the same env over JVM + gRPC, during training       11 steps/s  90.9 ms/step  per worker

so a single port process already beats the whole 16-JVM training run (175 steps/s) by ~27x.
That changes the right default: `DummyVecEnv` in one process is enough for most runs, and
`SubprocVecEnv` is for when you want more than ~4,800 steps/s.

The envs are otherwise unchanged -- same `JobPlacementEnv`, same observation assembly, reach
matrix, action mask and level sampling. Only the client differs, and
`benchmark/tools/diff_port.py` verifies that client step-for-step against the live gateway
(observations, reach rows, action masks, all nine ledger keys and terminated), 66 episodes with
zero problems.

    python3 benchmark/port_vec_env.py            # benchmark Dummy vs Subproc at several widths
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "tools"), os.path.join(os.path.dirname(HERE), "common/rl-manager")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def make_port_env(rank: int, params: dict, num_cpu: int, jobs_json: str = "[]"):
    """One env factory, picklable by name for SubprocVecEnv's spawn start method.

    Mirrors what `utils.misc` does for the gRPC path: build the env, then give it the member's
    level stream so every episode (including SB3's auto-resets) plays a fresh level from the
    configured split, with the worker's own deterministic share.
    """
    def _init():
        from gym_cloudsimplus.envs.job_placement import JobPlacementEnv
        from port_client import PortClient
        from utils.misc import level_stream

        env = JobPlacementEnv(params, jobs_as_json=jobs_json, client=PortClient())
        source, sampler = level_stream(params, rank)
        env.set_level_stream(source, sampler)
        return env

    return _init


def make_port_vec_env(params: dict, num_cpu: int = 1, subproc: bool | None = None):
    """A VecEnv of port-backed envs.

    subproc=None picks for you: one process while that is fast enough (num_cpu <= 4), processes
    above that. DummyVecEnv keeps everything in one process -- no pickling, no spawn, and the
    whole run is debuggable in a single stack trace.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    if subproc is None:
        subproc = num_cpu > 4
    fns = [make_port_env(i, params, num_cpu) for i in range(num_cpu)]
    venv = SubprocVecEnv(fns, start_method="spawn") if subproc else DummyVecEnv(fns)
    return VecMonitor(venv)


def _bench():
    import time

    import numpy as np

    import diff_port as dp

    params = dp.member_params("S", "test")
    print(f"{'config':28s} {'steps/s':>10s} {'vs 1 JVM worker':>16s} {'vs the 16-JVM run':>18s}")
    for width, subproc in ((1, False), (4, False), (8, True), (16, True)):
        venv = make_port_vec_env(params, num_cpu=width, subproc=subproc)
        try:
            venv.reset()
            k = venv.get_attr("max_jobs_waiting")[0]
            a = np.zeros((width, k), dtype=np.int64)
            for _ in range(10):
                venv.step(a)                                   # warm up spawn / first level
            n, t0 = 0, time.perf_counter()
            while n < 60:
                venv.step(a)
                n += 1
            dt = time.perf_counter() - t0
            sps = n * width / dt
            tag = f"{width:2d} env {'subproc' if subproc else 'dummy  '}"
            print(f"{tag:28s} {sps:10,.0f} {sps/11:15.0f}x {sps/175:17.0f}x")
        finally:
            venv.close()


if __name__ == "__main__":
    _bench()

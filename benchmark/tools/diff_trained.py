"""Does the Python port still match the jar under a TRAINED policy's actions?

benchmark/tools/diff_port.py verifies the port against the live gateway step for step --
observations, reach rows, action masks, all nine ledger keys, terminated -- but only under five
hand-written policies: random, defer, origin, cloud and scheduled. None of them visits the action
distribution a trained agent visits. A trained policy concentrates on a handful of (job, DC) pairs,
keeps DCs near saturation, and reaches states the scripted policies rarely produce. If the port
diverges anywhere, that is where it would show, and it is exactly the regime a paper's numbers
come from.

This replays a checkpoint's DETERMINISTIC policy -- `utils.evaluation.model_predictor`, the same
rule the val sweeps and `mode: evaluate` use, highest legal logit with ties broken by DC name --
choosing each action from the LIVE gateway's observation, and feeds that identical action to both
the gateway env and a port-backed env. Everything observable is compared at every step.

    python3 benchmark/tools/diff_trained.py <checkpoint.zip> [--member S] [--split test]
                                            [--levels 3]

Exit status 1 on any mismatch. Writes benchmark/results/diff_trained.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)
for _p in (BENCH, HERE, os.path.join(ROOT, "common/rl-manager")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diff_port as dp  # noqa: E402


def _numpy2_pickle_shim() -> None:
    """Checkpoints are written inside the container (numpy 2.x) and read here (numpy 1.26).

    numpy 2 moved its C core to `numpy._core`, and cloudpickle stores that module path inside the
    saved objects, so loading raises ModuleNotFoundError: numpy._core.numeric. Aliasing the new
    names onto the installed ones lets the pickle resolve; it is a read path only, and the arrays
    themselves are plain buffers. Harmless when numpy 2 is installed, since the names exist."""
    import numpy.core  # noqa: F401
    for name in list(sys.modules):
        if name == "numpy.core" or name.startswith("numpy.core."):
            sys.modules.setdefault("numpy._core" + name[len("numpy.core"):], sys.modules[name])

    # numpy 2 pickles a Generator's bit generator as the CLASS; numpy 1.26's ctor only accepts
    # its name and raises "is not a known BitGenerator module". Accept both. Gym spaces carry a
    # Generator, so this is on the path of every checkpoint load.
    import numpy.random._pickle as _np_pickle
    _orig = _np_pickle.__bit_generator_ctor

    def _ctor(bit_generator=None, *args, **kw):
        if isinstance(bit_generator, type):
            return bit_generator()
        return _orig(bit_generator, *args, **kw) if bit_generator is not None else _orig()

    _np_pickle.__bit_generator_ctor = _ctor
    for mod in ("numpy.random._generator", "numpy.random.mtrand"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "__bit_generator_ctor"):
            m.__bit_generator_ctor = _ctor

OBS_KEYS = ("infrastructure_state", "jobs_waiting_state", "reach_mask")
LEDGER_KEYS = dp.LEDGER_KEYS


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compare_level(model, params: dict, jobs_json: str, gateway) -> dict:
    """One level through both backends on the trained policy's own actions."""
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from utils.evaluation import model_predictor

    from port_client import PortClient

    jar_env = JobPlacementEnv(params, jobs_as_json="[]", port=gateway.port)
    port_env = JobPlacementEnv(params, jobs_as_json="[]", client=PortClient())

    # model_predictor wants a vec env only for its shapes; the actions are chosen from the
    # gateway's observation and then given to BOTH envs, so the trace is identical by construction
    predict = model_predictor(model, DummyVecEnv([lambda: jar_env]))

    o_jar, _ = jar_env.reset(options={"jobs_json": jobs_json})
    o_port, _ = port_env.reset(options={"jobs_json": jobs_json})
    problems, steps, r_jar, r_port = [], 0, 0.0, 0.0
    acted = np.zeros(jar_env.max_datacenters + 1, dtype=np.int64)

    while True:
        for k in OBS_KEYS:
            if not np.array_equal(o_jar[k], o_port[k]):
                problems.append(f"step {steps}: obs[{k!r}] differs")
        m_jar = np.asarray(jar_env.action_masks())
        m_port = np.asarray(port_env.action_masks())
        if not np.array_equal(m_jar, m_port):
            problems.append(f"step {steps}: action masks differ")

        batch = {k: np.asarray(o_jar[k])[None] for k in o_jar}
        action = np.asarray(predict(batch, m_jar[None])).reshape(-1)
        for a in action:
            acted[int(a)] += 1

        o_jar, _, t_jar, tr_jar, i_jar = jar_env.step(action)
        o_port, _, t_port, tr_port, i_port = port_env.step(action)
        steps += 1
        r_jar += i_jar["unshaped_reward"]
        r_port += i_port["unshaped_reward"]
        for key in LEDGER_KEYS:
            if i_jar[key] != i_port[key]:
                problems.append(f"step {steps}: {key} jar {i_jar[key]!r} port {i_port[key]!r}")
        if bool(t_jar) != bool(t_port) or bool(tr_jar) != bool(tr_port):
            problems.append(f"step {steps}: terminated jar {t_jar}/{tr_jar} port {t_port}/{tr_port}")
        if t_jar or t_port or tr_jar or tr_port:
            break

    nonzero = int(acted[1:].sum())
    return {"steps": steps, "return_jar": r_jar, "return_port": r_port,
            "return_residual": abs(r_jar - r_port), "problems": problems,
            "placements": nonzero, "noops": int(acted[0]),
            "distinct_dcs_used": int((acted[1:] > 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--member", default="S")
    ap.add_argument("--split", default="test")
    ap.add_argument("--levels", type=int, default=3)
    a = ap.parse_args()

    from sb3_contrib import MaskablePPO

    import run_references as rr
    import utils.levels as levels

    params = dp.member_params(a.member, a.split)
    src = levels.LevelSource(rr.MANIFEST, a.member,
                             [dc["name"] for dc in params["datacenters"]])

    class _A:
        plan_workers, plan_cache = 2, None

    ids = sorted(rr.prepare_levels({a.member: params}, a.split, False, _A())[a.member])[:a.levels]
    _numpy2_pickle_shim()
    # The spaces are rebuilt from params rather than unpickled: a gym Space carries a numpy
    # Generator whose state cannot cross the numpy 1 / 2 boundary, and SB3 then drops the spaces
    # entirely. They are a pure function of params, so rebuilding is exact, not a workaround.
    # The schedules are serialised closures and are not needed to act.
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    from port_client import PortClient
    _probe = JobPlacementEnv(params, jobs_as_json="[]", client=PortClient())
    model = MaskablePPO.load(a.checkpoint, device="cpu", custom_objects={
        "observation_space": _probe.observation_space,
        "action_space": _probe.action_space,
        "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})

    gateway = dp.Gateway()
    out = {"checkpoint": a.checkpoint, "checkpoint_sha256": _sha(a.checkpoint),
           "jar_sha256": dp.jar_sha256() if hasattr(dp, "jar_sha256") else None,
           "member": a.member, "split": a.split, "levels": {}}
    try:
        for lid in ids:
            rep = compare_level(model, params, src.jobs_json(lid), gateway)
            out["levels"][str(lid)] = rep
            flag = "OK " if not rep["problems"] else "BAD"
            print(f"  {flag} level {lid}: {rep['steps']} steps, "
                  f"return jar {rep['return_jar']:+.6f} port {rep['return_port']:+.6f} "
                  f"(residual {rep['return_residual']:.2e}), "
                  f"{rep['placements']} placements over {rep['distinct_dcs_used']} DCs, "
                  f"{rep['noops']} no-ops", flush=True)
            for p in rep["problems"][:5]:
                print(f"       {p}")
    finally:
        gateway.close()

    bad = sum(len(v["problems"]) for v in out["levels"].values())
    out["problems_total"] = bad
    out["max_return_residual"] = max(v["return_residual"] for v in out["levels"].values())
    os.makedirs(os.path.join(BENCH, "results"), exist_ok=True)
    with open(os.path.join(BENCH, "results", "diff_trained.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{len(out['levels'])} levels, {bad} problems, "
          f"max return residual {out['max_return_residual']:.3e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""A CloudSimGrpcClient work-alike backed by benchmark/cloudsim_port.py -- no JVM, no gRPC, no Docker.

`JobPlacementEnv` never touches the simulator directly: `CloudSimBaseEnv` only calls
`self._client.{create_simulation, reset, step, close, close_channel, ping}`. So the whole env --
observation assembly, reach matrix, action mask, step info -- works unchanged against any object
with that interface. This is that object, driving `cloudsim_port.Episode` in-process.

Why bother: measured on member S, 200-step level, same machine --

    cloudsim_port.Episode      16,103 steps/s   (one process)
    JVM + gRPC + Docker           130 steps/s   per worker, reference pass
    JVM + gRPC + Docker            11 steps/s   per worker, during PPO training

about 124x per worker, before any parallelism, and the port needs no port allocation, no
subprocess, no container and no JVM warm-up -- so a vector env can live in one process.

FIDELITY. benchmark/tools/diff_port.py replays the port and a LIVE gateway side by side, step for
step, on the same actions, and compares far more than the returns: the job-slot fingerprints
(jobs_waiting_state vs jobs_obs), the reach-mask rows, the ACTION MASKS, the host rows
(infrastructure_state vs infra_obs), all nine LEDGER_KEYS, and terminated/truncated. Stored
reports (benchmark/results/diff_port*.json, pinned to a jar_sha256 and a port_version):

    66 episodes, 0 with problems, 78,216 jobs finished, 5 policies
    max_return_residual 0.0   max_finish_residual 0.0   max_start_residual 0.0
    max_backlog_residual 0    finish_mismatches 0       vm_mismatches 0

So the observation the policy sees is verified equal, not just the score. The one known divergence
is documented as rule P9: once every job has finished, CloudSim can run out of events inside the
last step and shut down, and the jar then lists no hosts while the port always returns the rows.
diff_port detects that case and skips it; `_final_obs_empty_hosts` reproduces it here so the
behaviour is explicit and switchable rather than silently different.

The residual risk is coverage, not correctness: the five policies are random / defer / origin /
cloud / scheduled. A trained policy visits a different action distribution, so the honest extension
is to replay one trained episode's recorded actions through diff_port before trusting long runs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cloudsim_port as cp  # noqa: E402


class PortClient:
    """Duck-typed CloudSimGrpcClient over cloudsim_port.Episode.

    One instance holds one simulation, keyed by a sim id like the gRPC client, so a vector env
    can hold N of these in one process instead of N JVMs on N ports.
    """

    def __init__(self, final_obs_empty_hosts: bool = True):
        self._eps: dict[str, cp.Episode] = {}
        self._args: dict[str, tuple] = {}
        self._n = 0
        # match the jar's last-observation behaviour (P9); set False to keep the port's rows
        self._final_obs_empty_hosts = final_obs_empty_hosts

    # ---- lifecycle -----------------------------------------------------------------
    def create_simulation(self, params_json: str, jobs_json: str, rl_problem: str = None) -> str:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        self._n += 1
        sim_id = f"port-{self._n}"
        self._args[sim_id] = (params, jobs_json)
        self._eps[sim_id] = cp.Episode(params, jobs_json)
        return sim_id

    def reset(self, sim_id: str, seed: int = None, rl_problem: str = None, **kw) -> dict:
        params, jobs = self._args[sim_id]
        if "jobs_json" in kw and kw["jobs_json"] is not None:      # a fresh RING-N level
            jobs = kw["jobs_json"]
            self._args[sim_id] = (params, jobs)
        self._eps[sim_id] = cp.Episode(params, jobs)
        return {"observation": self._obs(sim_id)}

    def close(self, sim_id: str):
        self._eps.pop(sim_id, None)
        self._args.pop(sim_id, None)

    def close_channel(self):
        self._eps.clear()
        self._args.clear()

    def ping(self) -> bool:
        return True

    # ---- stepping ------------------------------------------------------------------
    def step(self, sim_id: str, action, rl_problem: str = None) -> dict:
        ep = self._eps[sim_id]
        led = ep.step(np.asarray(action, dtype=np.int64))   # advance() always returns the sums
        done = bool(led["terminated"])                      # the port's own termination flag
        return {
            "observation": self._obs(sim_id, final=done),
            "reward": float(led["unshaped_reward"]),         # shaping, if any, is the env's job
            "terminated": done,
            "truncated": False,                              # episodes terminate, never truncate
            "info": dict(led),
        }

    def batch_step(self, sim_id: str, actions, rl_problem: str = None) -> list:
        return [self.step(sim_id, a, rl_problem) for a in actions]

    # ---- wire format ---------------------------------------------------------------
    def _obs(self, sim_id: str, final: bool = False) -> dict:
        """The two int arrays the gateway sends: `JobPlacementEnv._get_observation` reads exactly
        these keys and does the rest (strip location, build reach_mask, pad)."""
        ep = self._eps[sim_id]
        infra = [] if (final and self._final_obs_empty_hosts) else list(ep.infra_obs())
        return {"infrastructure_observation": infra,
                "secondary_observation": list(ep.jobs_obs())}

    # `advance()` already emits sla_value_realized, sla_penalty_paid, resource_cost, jobs_met,
    # jobs_violated, jobs_expired_unplaced, jobs_waiting, jobs_placed, offered_value,
    # unshaped_reward, terminated and step -- a superset of _STEP_INFO_KEYS except
    # jobs_placed_ratio, job_wait_time and potential, which JobPlacementEnv derives or defaults.

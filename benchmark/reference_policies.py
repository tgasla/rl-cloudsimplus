"""RING-N reference policies R0-R4, played through the same observation, action space and action
mask as the RL agents (spec: refpol_spec.md §3-5).

  R0     random-feasible: per real slot, uniform over [no-op] + the legal DCs (the floor G_rand)
  R1     earliest-shortest-to-most-free-dc: always binds, to the legal DC with most free PEs,
         then the least backlog
  R2     earliest-most-critical-to-nearest-dc: nearest legal DC with a host that has the PEs
         free now (origin, ring neighbours, cloud); otherwise the job waits
  R3     value-density-to-fastest-feasible (margin 0); R3-f1 margin 1; R3-c0 / R3-c1 prefer the
         cheapest feasible DC
  R4-CDLS  plays the clairvoyant plan benchmark/r4_planner.py made for the level (R4Executor);
         the runner's R4 row is the best of R4-CDLS and R1-R3 per level

R1 and R2 are the gateway's rule-based cloudlet_to_dc_mapping policies (WrappedSimulation), which
decide on the same window and observation and place through the same action path: on RING-N
levels the two sides place every job alike and score identical returns
(test_reference_policies.test_r1_and_r2_score_as_the_java_rules_do).

A policy is built per worker: act(view, ctx) -> int[n_slots], with begin_episode(ctx, record)
called at the first decision of every episode. ReferencePredictor is the predict(obs, masks)
callable utils.evaluation.play_levels expects; it checks every action against the mask.

An action names a DC only: the simulator binds the step's placements in slot order, each to the
most-free VM of its DC, while a policy's in-step bookkeeping follows its own job order. Two jobs
placed on one DC can therefore swap hosts, which only permutes identical hosts unless a job
then queues (measured on S, test split: under 0.1% of R2 placements, under 1% of R3's).
"""
import hashlib
import math
from dataclasses import dataclass
from functools import partial

import numpy as np

import cloudsim_port as cp
from refpol_common import OBS_KEYS, ObsView, Topology

R0_SEED = 20260929


class IllegalActionError(RuntimeError):
    pass


class PlanDivergence(RuntimeError):
    """R4-CDLS in strict mode: the gateway did not do what the plan predicted."""


@dataclass
class Ctx:
    step: int              # the worker env's _current_step: steps taken so far this episode
    level_id: int | None
    rollout: int
    worker: int
    levels: object         # the worker's LevelSource (None outside RING-N)
    strict: bool


class Policy:
    name = None

    def __init__(self, topo: Topology, **_):
        self.topo = topo

    def begin_episode(self, ctx: Ctx, record: dict) -> None:
        """Called at step 0 of every episode; `record` becomes extra columns of its row."""

    def act(self, view: ObsView, ctx: Ctx) -> np.ndarray:
        raise NotImplementedError

    def observe(self, info: dict, record: dict) -> None:
        """The info of the step just taken (runner and tests pass it on through InfoTap)."""


class R0(Policy):
    """Mask-uniform random: the draw indexes [0] + the legal DCs in static-key order, so it
    names the same DC on a relabelled topology. The stream is (R0_SEED, level, rollout)."""
    name = "R0"

    def begin_episode(self, ctx, record):
        if ctx.level_id is None:
            raise ValueError("R0 needs a level id (RING-N levels)")
        self.rng = np.random.default_rng([R0_SEED, int(ctx.level_id), int(ctx.rollout)])

    def act(self, view, ctx):
        actions = np.zeros(self.topo.n_slots, dtype=np.int64)
        for i in view.slots:
            choices = [0] + sorted(view.legal(i), key=self.topo.static_key)
            actions[i] = choices[self.rng.integers(len(choices))]
        return actions


class R1(Policy):
    """Jobs by (ttd, r, slot); each goes to the legal DC with the most free PEs, then the least
    backlog, then the static key. A DC's free PEs are the sum of its hosts' observed free_pes (0
    at the least), a job placed this step using its cores on the DC's host with the most free
    PEs (the first such host); its backlog is the sum of its hosts' backlog_core_ts plus the
    core-timesteps placed there this step (cores * r * mips_ref / that host's MIPS). Always
    binds: a full DC queues the job. The gateway's R1 is the same rule, float for float
    (WrappedSimulation.executeEarliestShortestCloudletToMostFreeDcAction; its (due, length) job
    order is this (ttd, r) order on RING-N, whose due times are whole timesteps and whose mi are
    multiples of mips_ref)."""
    name = "R1"

    def act(self, view, ctx):
        topo = self.topo
        actions = np.zeros(topo.n_slots, dtype=np.int64)
        free = {k: list(view.free[k]) for k in topo.ks}
        backlog = {k: sum(view.backlog[k]) for k in topo.ks}           # ints: exact
        placed_work = dict.fromkeys(topo.ks, 0.0)
        for i in sorted(view.slots, key=lambda i: (view.ttd[i], view.r[i], i)):
            legal = view.legal(i)
            if not legal:
                continue
            k = min(legal, key=lambda k: (-sum(free[k]), backlog[k] + placed_work[k])
                    + topo.static_key(k))
            hosts = free[k]
            h = max(range(len(hosts)), key=lambda h: (hosts[h], -h))
            hosts[h] = max(0, hosts[h] - view.cores[i])
            # the Java expression, cores * r * mips_ref / mips, in the same order
            placed_work[k] += view.cores[i] * view.r[i] * topo.mips_ref / topo.dc(k).vm_mips[h]
            actions[i] = k
        return actions


class R2(Policy):
    """Jobs by (ttd, sensitivity desc, slot); legal DCs by hops from the origin (0 itself,
    2 the cloud, 1 any other) then static key; the job goes to the first whose host with the
    most free PEs (observed free_pes less this step's placements; the first such host) holds
    it, else waits (no-op). The gateway's R2 is the same rule
    (WrappedSimulation.executeEarliestMostCriticalCloudletToNearestDcAction)."""
    name = "R2"

    def __init__(self, topo, **kwargs):
        super().__init__(topo, **kwargs)
        # The observation strips the origin; each origin has its own reach set.
        by_set = {}
        for k in topo.ks:
            by_set.setdefault(topo.legal_set(k), []).append(k)
        shared = [[topo.dc(k).name for k in ks] for ks in by_set.values() if len(ks) > 1]
        if shared:
            raise ValueError(f"R2 cannot tell these origins apart by their reach: {shared}")
        self.origin_of = {s: ks[0] for s, ks in by_set.items()}

    def hops(self, origin: int, k: int) -> int:
        if k == origin:
            return 0
        return 2 if self.topo.dc(k).type == "cloud" else 1

    def act(self, view, ctx):
        topo = self.topo
        actions = np.zeros(topo.n_slots, dtype=np.int64)
        free = {k: list(view.free[k]) for k in topo.ks}
        for i in sorted(view.slots, key=lambda i: (view.ttd[i], -view.sens[i], i)):
            origin = self.origin_of[view.reach_set(i)]
            for k in sorted(view.legal(i), key=lambda k: (self.hops(origin, k), topo.static_key(k))):
                hosts = free[k]
                h = max(range(len(hosts)), key=lambda h: (hosts[h], -h))
                if hosts[h] >= view.cores[i]:
                    actions[i] = k
                    hosts[h] -= view.cores[i]
                    break
        return actions


class R3(Policy):
    """Value density to the preferred feasible DC.

    Per legal DC k with cost < V + P: the host with (most free PEs, least backlog, lowest
    index) among those big enough; wait 0 if it has the PEs free, else backlog / max(vm_pes -
    cores, 1); ECT = network delay + wait + runtime. k is feasible if ECT + margin <= ttd.
    Jobs go by (density desc, min ECT at step start, slot); "fastest" picks the feasible DC by
    (ECT, cost, dyn key), "cheapest" by (cost, ECT, dyn key); a job with none waits.
    """
    NAMES = {("fastest", 0): "R3", ("fastest", 1): "R3-f1",
             ("cheapest", 0): "R3-c0", ("cheapest", 1): "R3-c1"}

    def __init__(self, topo, pref: str = "fastest", margin: int = 0, **kwargs):
        super().__init__(topo, **kwargs)
        self.name = self.NAMES[(pref, margin)]
        self.pref, self.margin = pref, margin

    def act(self, view, ctx):
        topo = self.topo
        actions = np.zeros(topo.n_slots, dtype=np.int64)
        free = {k: list(view.free[k]) for k in topo.ks}
        back = {k: [float(b) for b in view.backlog[k]] for k in topo.ks}
        dc_free = {k: sum(free[k]) for k in topo.ks}

        def estimate(i, k):
            """(ECT, cost, host, runtime) of slot i on DC k, or None if it costs >= V + P."""
            cost = view.cost(i, k)
            if cost >= view.V[i] + view.P[i]:
                return None
            cores, dc = view.cores[i], topo.dc(k)
            fits = [h for h, pes in enumerate(dc.vm_pes) if pes >= cores]
            h = max(fits, key=lambda h: (free[k][h], -back[k][h], -h))
            wait = 0.0 if free[k][h] >= cores else back[k][h] / max(dc.vm_pes[h] - cores, 1)
            run = view.run(i, k, h)
            return dc.nd + wait + run, cost, h, run

        min_ect0 = {}
        for i in view.slots:
            ects = [e[0] for e in (estimate(i, k) for k in view.legal(i)) if e is not None]
            min_ect0[i] = min(ects, default=float("inf"))
        for i in sorted(view.slots, key=lambda i: (-view.density(i), min_ect0[i], i)):
            best = None
            for k in view.legal(i):
                e = estimate(i, k)
                if e is None:
                    continue
                ect, cost, h, run = e
                if ect + self.margin > view.ttd[i]:
                    continue
                key = ((ect, cost) if self.pref == "fastest" else (cost, ect)) \
                    + topo.dyn_key(k, dc_free[k])
                if best is None or key < best[0]:
                    best = (key, k, h, run)
            if best is None:
                continue
            _, k, h, run = best
            cores = view.cores[i]
            actions[i] = k
            free[k][h] -= cores
            back[k][h] += cores * run
            dc_free[k] -= cores
        return actions


class R4Executor(Policy):
    """R4-CDLS: plays the plan r4_planner made for the level (spec §5C), with the plan's
    predictions checked at every step.

    plans: {level id: plan}. At step 0 the level's jobs must hash to the plan's jobs_sha and no
    real slot may be visible. At every step each planned slot's fingerprint (cores, r, ttd,
    sensitivity, reach row, computed from the level's jobs) must match the observation, and so
    must every host's free_pes (exactly) and backlog_core_ts (within 1); the step's ledger (the
    info observe() gets) must be the predicted one. Any mismatch is a divergence: strict mode
    raises PlanDivergence, benchmark mode records the step (r4_divergent_steps,
    r4_first_divergence_step) and plays only the planned actions of slots whose fingerprint
    matches. The record also gets the predicted return, r4_pred_return.
    """
    name = "R4-CDLS"

    def __init__(self, topo, plans=None, strict=False, **kwargs):
        super().__init__(topo, **kwargs)
        self.plans = plans or {}
        self.strict = strict

    def begin_episode(self, ctx, record):
        if ctx.level_id is None or int(ctx.level_id) not in self.plans:
            raise KeyError(f"no R4 plan for level {ctx.level_id}")
        self.level = int(ctx.level_id)
        self.plan = self.plans[self.level]
        jobs_json = ctx.levels.jobs_json(self.level)
        if hashlib.sha256(jobs_json.encode()).hexdigest() != self.plan["jobs_sha"]:
            raise PlanDivergence(f"level {self.level}: its jobs are not the ones the plan was made for")
        self.cfg = cp.Settings(self.topo.params)
        self.jobs = {job.id: job for job in cp.parse_jobs(jobs_json, self.cfg)}
        self.record = record
        self.last_step = None
        self.divergent = set()
        record.update(r4_pred_return=self.plan["return"], r4_divergent_steps=0,
                      r4_first_divergence_step=None)

    def _diverge(self, s: int, what: str) -> None:
        if self.strict:
            raise PlanDivergence(f"level {self.level} step {s}: {what}")
        self.divergent.add(s)
        self.record.update(r4_divergent_steps=len(self.divergent),
                           r4_first_divergence_step=min(self.divergent))

    def fingerprint(self, job_id: int, clock: float) -> tuple:
        """(cores, r, ttd, sensitivity, reach set) of a job's slot at a step starting at clock,
        as CloudSimProxy.getJobsWaitingObservation and JobPlacementEnv encode it."""
        job = self.jobs[job_id]
        ttd = int(max(0.0, math.floor((job.due - clock) / self.cfg.interval)))
        return job.cores, job.r, ttd, job.sens, self.topo.legal_set(job.loc + 1)

    def act(self, view, ctx):
        s = int(ctx.step)
        step = self.plan["steps"][s]
        if s == 0 and view.slots:
            self._diverge(s, f"{len(view.slots)} jobs visible at step 0")
        clock = self.cfg.step_clock(s)
        matched = set()
        for i, job_id in enumerate(step["visible"]):
            if view.real[i] and self.fingerprint(job_id, clock) == (
                    view.cores[i], view.r[i], view.ttd[i], view.sens[i], view.reach_set(i)):
                matched.add(i)
            else:
                self._diverge(s, f"slot {i} is not job {job_id}")
        if len(view.slots) != len(step["visible"]):
            self._diverge(s, f"{len(view.slots)} jobs visible, the plan has {len(step['visible'])}")
        free = [f for k in self.topo.ks for f in view.free[k]]
        backlog = [b for k in self.topo.ks for b in view.backlog[k]]
        if free != step["free"]:
            self._diverge(s, "free_pes differ")
        if any(abs(a - b) > 1 for a, b in zip(backlog, step["backlog"])):
            self._diverge(s, "backlog_core_ts differ by more than 1")
        actions = np.zeros(self.topo.n_slots, dtype=np.int64)
        for i, k in step["actions"]:
            if i in matched:
                actions[i] = k
        self.last_step = s
        return actions

    def observe(self, info, record):
        s = self.last_step
        predicted = self.plan["steps"][s]["ledger"]
        wrong = [key for key, value in predicted.items() if info[key] != value]
        if wrong:
            self._diverge(s, "ledger " + ", ".join(f"{key} {info[key]!r} (plan {predicted[key]!r})"
                                                   for key in wrong))


POLICIES = {
    "R0": R0,
    "R1": R1,
    "R2": R2,
    "R3": partial(R3, pref="fastest", margin=0),
    "R3-f1": partial(R3, pref="fastest", margin=1),
    "R3-c0": partial(R3, pref="cheapest", margin=0),
    "R3-c1": partial(R3, pref="cheapest", margin=1),
    "R4-CDLS": R4Executor,
}
R3_FAMILY = ("R3", "R3-f1", "R3-c0", "R3-c1")
R4_POLICY = "R4-CDLS"
# The policies that decide from the observation alone; R4-CDLS needs its plans.
ONLINE_POLICIES = ("R0", "R1", "R2") + R3_FAMILY


def make_policy(name: str, topo: Topology, **kwargs) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown reference policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](topo, **kwargs)


def check_actions(actions: np.ndarray, view: ObsView) -> None:
    """Hard error on an action outside the mask; padding slots must take the no-op."""
    slots = np.arange(view.topo.n_slots)
    if actions.shape != slots.shape or (actions < 0).any() or (actions >= view.topo.n_actions).any():
        raise IllegalActionError(f"actions out of range: {actions}")
    bad = ~view.mask[slots, actions] | (~view.real & (actions != 0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise IllegalActionError(f"slot {i}: action {actions[i]} is not in its mask row "
                                 f"{np.flatnonzero(view.mask[i]).tolist()}")


class ReferencePredictor:
    """predict(obs, masks) -> actions [n_envs, n_slots] for utils.evaluation.play_levels.

    One policy instance per worker, on that worker's topology (params). The worker env's
    _current_step marks the episode start; `episodes[w]` holds one record per episode worker w
    started, in order (the rows of play_levels for w are its first episodes, in the same order).
    A record carries the level id, the window statistics (decisions; window_full_steps: steps
    with every slot taken; deferred_slots: real slots left with the no-op; hidden_jobs, filled
    only if the step infos reach observe(): arrived jobs outside the window, summed over steps)
    and whatever the policy adds.
    """

    def __init__(self, vec_env, policy: str, rollout: int = 0, strict: bool = False,
                 **policy_kwargs):
        self.vec_env = vec_env
        self.policy = policy
        self.rollout = rollout
        self.strict = strict
        self.levels = vec_env.get_attr("_levels")
        self.topos = [Topology.from_params(p) for p in vec_env.get_attr("params")]
        self.policies = [make_policy(policy, topo, strict=strict, **policy_kwargs)
                         for topo in self.topos]
        self.episodes = [[] for _ in self.topos]

    def __call__(self, obs, masks):
        steps = self.vec_env.get_attr("_current_step")
        level_ids = self.vec_env.get_attr("_level_id")
        actions = np.zeros((len(self.topos), self.topos[0].n_slots), dtype=np.int64)
        for w, topo in enumerate(self.topos):
            ctx = Ctx(step=steps[w], level_id=level_ids[w], rollout=self.rollout, worker=w,
                      levels=self.levels[w], strict=self.strict)
            if steps[w] == 0:
                record = {"level_id": level_ids[w], "decisions": 0, "window_full_steps": 0,
                          "deferred_slots": 0, "hidden_jobs": 0}
                self.episodes[w].append(record)
                self.policies[w].begin_episode(ctx, record)
            elif not self.episodes[w]:
                raise RuntimeError(f"worker {w} is mid-episode; reset the env first")
            view = ObsView({key: obs[key][w] for key in OBS_KEYS}, masks[w], topo)
            a = np.asarray(self.policies[w].act(view, ctx), dtype=np.int64)
            check_actions(a, view)
            record = self.episodes[w][-1]
            record["decisions"] += 1
            record["window_full_steps"] += int(view.real.all())
            record["deferred_slots"] += int((view.real & (a == 0)).sum())
            actions[w] = a
        return actions

    def observe(self, infos) -> None:
        """The infos of the step just taken. Their jobs_waiting counts every arrived, unplaced
        job at the decision (visible or not), so the excess over the window was hidden."""
        for w, info in enumerate(infos):
            record = self.episodes[w][-1]
            record["hidden_jobs"] += max(0, int(info["jobs_waiting"]) - self.topos[w].n_slots)
            self.policies[w].observe(info, record)

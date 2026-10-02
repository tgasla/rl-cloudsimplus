"""R4-CDLS, the clairvoyant reference policy of the RING-N benchmark (refpol_spec.md §5B-5D).

A clairvoyant density list scheduler with exact simulation. The only future information it
uses is the level's job list (jobs_json); it plans on benchmark/cloudsim_port.py, the exact
replica of the gateway jar, so a plan's predicted return is what the jar pays for its actions.

Phase A, density list admission. Jobs go in (value density (V + P) / (cores * r) desc, due,
arrival, id) order; each takes one DC and bind step, or none. Per legal DC with cost c < V + P,
the candidate bind steps are its arrival step, then the first CANDIDATE_STEPS distinct
ceil(execution end) of the jobs committed on that DC, up to the last step from which it could
still be on time. A step is taken if it passes
  the queue check: the job is among the first max_jobs_waiting of the pool at that step, and
      keeping it in the pool until then pushes no committed job that is in the window without
      it out of the window at that job's bind step. Pool intervals: committed [a, bind step],
      rejected [a, min(due, H - 1)], not yet planned [a, a] in iteration 1 and their Phase-B
      interval of iteration 1 in iteration 2;
  the exact check: the DC replayed from its state at that step with the job added to its
      committed binds, the job is on time (execution end + g <= due) and every committed job
      that is on time stays so (no harm).
  The first step that passes is the DC's candidate; the lowest placement cost wins (the job's
  net value V - c), then the earliest execution end, the bind step, (tier, capacity desc, name).
  A job with no candidate is rejected (never placed: -P, which beats -P - c), unless staying in
  the pool until its due would push committed jobs out of the window, which the queue check
  forbids a commit: it is then dumped, bound at its first visible step on the legal DC where
  it loses the least value (its cost, less its own V + P if it is on time there, plus V + P of
  every job it makes late), if that is less than the V + P of the jobs it would push out. The
  environment has no reject action: an unplaced job leaves the pool only at its due.
Phase B, exact chronological replay with repair: the plan played on the full port, step by step,
  as the executor plays it on the jar. A job binds at its planned step if it is visible then; a
  job hidden at its planned step is re-planned when it shows, on the DC that passes the exact
  check from the replay's state (with the remaining planned binds) in Phase A's choice order,
  or else dumped as in Phase A.
Two iterations; the plan is the Phase-B replay with the higher return.

zc_ceiling(params, jobs_json) is the zero-contention ceiling of spec §5D: every job alone on an
empty DC, bound at its arrival, at its best legal DC.
"""
from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
import os
import tempfile
import time
from itertools import islice

import cloudsim_port as cp

R4_GUARD_BAND = 0.0      # s; the port is bit-exact on start and finish times (results/diff_port*.json)
CANDIDATE_STEPS = 8      # K: later bind steps tried per DC after the arrival step
ITERATIONS = 2
PLAN_VERSION = "r4-cdls/1"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache", "r4")
LEDGER_KEYS = ("jobs_placed", "jobs_met", "jobs_violated", "jobs_expired_unplaced",
               "sla_value_realized", "sla_penalty_paid", "resource_cost", "unshaped_reward")
# What the port reads from params (cloudsim_port.Settings and DcSpec): part of a plan's cache key.
PARAM_KEYS = ("timestep_interval", "min_time_between_events", "max_episode_length",
              "max_jobs_waiting", "mips_ref", "cloudlet_to_dc_mapping", "cloudlet_to_vm_mapping",
              "split_large_jobs", "drift_at_step", "reward_shaping", "datacenters")
PARAM_PREFIXES = ("sla_value_", "sla_penalty_", "cost_", "network_delay_")


def density(job: cp.Job) -> float:
    return (job.V + job.P) / (job.cores * max(job.r, 1e-9))


def run_until_started(state: cp.DcState, s: int, binds: dict, track, must_meet=(),
                      g: float = R4_GUARD_BAND) -> tuple:
    """Run a DC until every job of `track` has started.

    state: the DC at the start of step s, before that step's binds; it is advanced in place.
    binds: {step: [job index]} bound to the DC from step s on. track: jobs not started yet.
    Returns (ok, started), started mapping each job of `track` that started to (start, vm). ok is
    False, and the run stops early, once a job of must_meet cannot be on time any more: it
    started too late (execution end + g > due), or a step ended before it started and a start
    at that time would already be too late.
    """
    cfg, jobs, mips = state.cfg, state.jobs, state.spec.vm_mips
    eps = cp.ON_TIME_EPSILON
    remaining = set(track)
    started = {}
    watch, border = [], []      # submitted jobs of must_meet not started yet: (latest start, job)

    def ends_late(j, start):
        job = jobs[j]
        return start + job.L / mips[state.vm_of[j]] + g > job.due + eps

    def watch_job(j):
        job = jobs[j]
        heapq.heappush(watch, (job.due + eps - g - job.L / mips[state.vm_of[j]], j))

    for j in remaining:
        if j in state.start:
            raise ValueError(f"job {jobs[j].id} has started already")
        if j in must_meet and j in state.vm_of:
            watch_job(j)
    c = cfg.step_clock(s)
    last_bind = max(binds, default=-1)
    while remaining:
        T = cfg.interval if s == 0 else c + cfg.interval
        subs = state.bind_submits(c, binds[s]) if s in binds else ()
        n_started = len(state.start)
        state.run_step(c, T, subs)
        new = len(state.start) - n_started
        if new:
            for j in islice(reversed(state.start), new):        # the step's starts
                if j in remaining:
                    remaining.discard(j)
                    start = state.start[j]
                    started[j] = (start, state.vm_of[j])
                    if j in must_meet and ends_late(j, start):
                        return False, started
        for j, _, _ in subs:
            if j in remaining and j in must_meet:
                watch_job(j)
        while watch and watch[0][0] < T + 1e-6:     # the heap key is approximate; ends_late is exact
            border.append(heapq.heappop(watch)[1])
        if border:
            border = [j for j in border if j in remaining]
            if any(ends_late(j, T) for j in border):  # starts at T at the earliest
                return False, started
        s += 1
        c = T
        if remaining and s > last_bind and not state.busy():
            raise RuntimeError(f"{state.spec.name}: jobs {sorted(jobs[j].id for j in remaining)} "
                               f"were never bound here")
    return True, started


def on_time(job: cp.Job, start: float, mips: float, g: float) -> bool:
    return start + job.L / mips + g <= job.due + cp.ON_TIME_EPSILON


def value_change(jobs: list, spec: cp.DcSpec, met: set, started: dict, g: float) -> float:
    """V + P gained for every job of `started` that is on time and was not in `met`, lost for
    every one that was and is not."""
    out = 0.0
    for k, (start, v) in started.items():
        now = on_time(jobs[k], start, spec.vm_mips[v], g)
        if now != (k in met):
            out += (jobs[k].V + jobs[k].P) * (1 if now else -1)
    return out


def unstarted(state: cp.DcState) -> set:
    """Jobs submitted to the DC that have not started: in flight or waiting."""
    out = set(state.inflight)
    for lst in state.wait:
        out.update(x.j for x in lst)
    return out


# ─── Phase A ─────────────────────────────────────────────────────────────────

class _DcPlan:
    """One DC in Phase A: its committed binds, its trajectory under them (the state at each step
    start, before that step's binds, computed on demand) and every committed job's start."""

    def __init__(self, spec: cp.DcSpec, jobs: list, cfg: cp.Settings, g: float):
        self.spec, self.jobs, self.cfg, self.g = spec, jobs, cfg, g
        idle = cp.DcState(spec, jobs, cfg)
        idle.step = 1                    # no bind of step 0 finds a VM: the DC is idle until step 1
        self.snaps = {1: idle}
        self.valid_to = 1                # snaps[1..valid_to] hold for the current binds
        self.binds = {}                  # step -> committed jobs bound at it
        self.placed = {}                 # committed job -> (start, vm) on the current trajectory
        self.met = set()                 # committed jobs on time on the current trajectory
        self.ends = []                   # distinct ceil(execution end) of committed jobs, sorted
        self._end_of, self._end_count = {}, {}

    def state_at(self, s: int) -> cp.DcState:
        if s <= self.valid_to:
            return self.snaps[s]
        t = self.valid_to
        state = self.snaps[t].copy()
        c = self.cfg.step_clock(t)
        while t < s:
            T = c + self.cfg.interval
            state.run_step(c, T, state.bind_submits(c, self.binds[t]) if t in self.binds else ())
            t, c = t + 1, T
            self.snaps[t] = state
            state = state.copy()
        self.valid_to = s
        return self.snaps[s]

    def check(self, j: int, s: int, require: bool = True) -> tuple:
        """The exact check of binding job j here at step s: (ok, started) of run_until_started
        on every committed job not started by then, and j, which must all be on time if they
        are met now (j: always). With require False nothing is required (a dump): the run
        goes on until every job has started."""
        state = self.state_at(s).copy()
        binds = {t: list(js) for t, js in self.binds.items() if t >= s}
        binds.setdefault(s, []).append(j)
        track = unstarted(state)
        for js in binds.values():
            track.update(js)
        must = ({k for k in track if k in self.met} | {j}) if require else ()
        return run_until_started(state, s, binds, track, must, self.g)

    def commit(self, j: int, s: int, started: dict) -> None:
        """Bind j at step s; started is its passing check's (every job not started by s)."""
        self.binds.setdefault(s, []).append(j)
        for t in [t for t in self.snaps if t > s]:
            del self.snaps[t]
        self.valid_to = min(self.valid_to, s)
        for k, (start, v) in started.items():
            self.placed[k] = (start, v)
            job = self.jobs[k]
            if on_time(job, start, self.spec.vm_mips[v], self.g):
                self.met.add(k)
            else:
                self.met.discard(k)
            end = math.ceil(start + job.L / self.spec.vm_mips[v])
            old = self._end_of.get(k)
            if old == end:
                continue
            if old is not None:
                self._end_count[old] -= 1
                if not self._end_count[old]:
                    del self._end_count[old]
                    del self.ends[bisect.bisect_left(self.ends, old)]
            self._end_of[k] = end
            if end not in self._end_count:
                self._end_count[end] = 0
                bisect.insort(self.ends, end)
            self._end_count[end] += 1

    def later_steps(self, lo: int, hi: int) -> list:
        """The first CANDIDATE_STEPS execution-end steps of committed jobs in (lo, hi]."""
        i = bisect.bisect_right(self.ends, lo)
        return [t for t in self.ends[i:i + CANDIDATE_STEPS] if t <= hi]


class _Pool:
    """Which jobs are in the pool (arrived, not bound, not evicted) at each step, as sorted
    (due, arrival, id) ranks; a job's interval is [lo, hi], empty when hi < lo."""

    def __init__(self, krank: list, H: int, spans):
        self.krank = krank
        self.at = [[] for _ in range(H)]
        self.span = [(0, -1)] * len(krank)
        for j, (lo, hi) in enumerate(spans):
            self.set(j, lo, hi)

    def set(self, j: int, lo: int, hi: int) -> None:
        hi = min(hi, len(self.at) - 1)
        old_lo, old_hi = self.span[j]
        r = self.krank[j]
        for t in range(old_lo, old_hi + 1):
            if not lo <= t <= hi:
                lst = self.at[t]
                del lst[bisect.bisect_left(lst, r)]
        for t in range(lo, hi + 1):
            if not old_lo <= t <= old_hi:
                bisect.insort(self.at[t], r)
        self.span[j] = (lo, hi)

    def rank(self, j: int, t: int) -> int:
        """How many jobs of the pool at step t come before j in the window."""
        return bisect.bisect_left(self.at[t], self.krank[j])

    def holds(self, j: int, t: int) -> bool:
        lo, hi = self.span[j]
        return lo <= t <= hi


class _Level:
    """The static facts of one level."""

    def __init__(self, params: dict, jobs_json, g: float):
        ep = cp.Episode(params, jobs_json)
        cfg = ep.cfg
        if cfg.interval != 1.0:
            raise ValueError("the planner counts bind steps in seconds: timestep_interval must be 1")
        self.params, self.jobs_json, self.g = params, jobs_json, g
        self.cfg, self.specs, self.jobs = cfg, ep.specs, ep.jobs
        self.H, self.n_slots = cfg.horizon, cfg.max_jobs_waiting
        jobs = self.jobs
        self.krank = [0] * len(jobs)
        for r, j in enumerate(sorted(range(len(jobs)),
                                     key=lambda j: (jobs[j].due, jobs[j].a, jobs[j].id))):
            self.krank[j] = r
        self.static = [(sp.type_id, -sp.capacity, sp.name) for sp in self.specs]
        self.legal = [cp.legal_dcs(job, self.specs) for job in jobs]
        self.order = sorted(range(len(jobs)), key=lambda j: (-density(jobs[j]), jobs[j].due,
                                                             jobs[j].a, jobs[j].id))

    def options(self, j: int, lo: int) -> tuple:
        """The DCs worth trying for job j from step lo, grouped by placement cost, ascending:
        [(cost, [(static key, d, latest bind step)])], and why there are none."""
        job, cfg = self.jobs[j], self.cfg
        groups, reason = {}, "no_dc" if not self.legal[j] else "cost"
        for d in self.legal[j]:
            spec = self.specs[d]
            cost = job.cost_on(spec, cfg)
            if cost >= job.V + job.P:
                continue
            run = min(job.L / spec.vm_mips[v] for v in range(spec.n_vms) if spec.vm_pes[v] >= job.cores)
            latest = min(self.H - 1, math.floor(job.due - self.g - spec.net_delay - run + 1e-6))
            if latest < lo:
                reason = "late"
                continue
            groups.setdefault(cost, []).append((self.static[d], d, latest))
        return [(cost, sorted(groups[cost])) for cost in sorted(groups)], reason

    def admits(self, pool: _Pool, bound_at: dict, j: int, s: int) -> bool:
        """The queue check of binding job j at step s."""
        return pool.rank(j, s) < self.n_slots and not self.pushes_out(pool, bound_at, j, s)

    def pushes_out(self, pool: _Pool, bound_at: dict, j: int, hi: int, first: bool = True) -> list:
        """The jobs committed earlier that keeping job j in the pool until step hi pushes out of
        the window at their bind step (each in the window's last slot there); only the first
        one found if `first`."""
        K, r = self.n_slots, self.krank[j]
        out = []
        for t in range(int(self.jobs[j].a) + 1, hi + 1):
            if pool.holds(j, t):
                continue
            for k in bound_at.get(t, ()):
                if self.krank[k] > r and pool.rank(k, t) == K - 1:
                    out.append(k)
                    if first:
                        return out
        return out


def _dump(lv: _Level, pool: _Pool, dcs: list, j: int, lo: int, hi: int, at_risk: float):
    """Where a job with no candidate is placed when keeping it in the pool until its due would
    push committed jobs worth at_risk (their V + P) out of the window: at the first step from lo
    it is visible, on the legal DC where placing it loses the least value (its cost, less its own
    V + P if it is on time there, plus V + P of every met job it makes late), then the cheapest,
    then the static key. (d, step, started), or None if it is never visible or that loses at
    least at_risk."""
    job = lv.jobs[j]
    for s in range(lo, hi + 1):
        if pool.rank(j, s) >= lv.n_slots:
            continue
        best = None
        for d in lv.legal[j]:
            spec = lv.specs[d]
            cost = job.cost_on(spec, lv.cfg)
            _, started = dcs[d].check(j, s, require=False)
            key = (cost - value_change(lv.jobs, spec, dcs[d].met, started, lv.g), cost, lv.static[d])
            if best is None or key < best[0]:
                best = (key, d, started)
        return (best[1], s, best[2]) if best[0][0] < at_risk else None
    return None


def _phase_a(lv: _Level, spans: list) -> tuple:
    """spans: each job's assumed pool interval until it is planned. Returns ({job: (dc, step)},
    {rejected job: reason}, {dumped job: reason}); the plan holds the dumped jobs too."""
    pool = _Pool(lv.krank, lv.H, spans)
    dcs = [_DcPlan(spec, lv.jobs, lv.cfg, lv.g) for spec in lv.specs]
    bound_at, plan, rejected, dumped = {}, {}, {}, {}
    for j in lv.order:
        job = lv.jobs[j]
        a = int(job.a)
        if a > lv.H - 1:
            rejected[j] = "horizon"                 # never visible
            continue
        lo = max(a, 1)                              # no bind of step 0 finds a VM
        groups, reason = lv.options(j, lo)
        choice = None
        for _, members in groups:
            found = []
            for key, d, latest in members:
                for s in [lo] + dcs[d].later_steps(lo, latest):
                    if not lv.admits(pool, bound_at, j, s):
                        continue
                    ok, started = dcs[d].check(j, s)
                    if ok:
                        start, v = started[j]
                        found.append((start + job.L / lv.specs[d].vm_mips[v], s, key, d, started))
                        break
            if found:
                choice = min(found, key=lambda f: f[:3])
                break
        if choice is None:
            reason = reason if not groups else "no_candidate"
            hi = min(math.floor(job.due), lv.H - 1)
            # Rejected, the job stays in the pool until its due; where that would push committed
            # jobs out of the window, which the queue check forbids a commit, it is dumped
            # unless that loses more.
            out = lv.pushes_out(pool, bound_at, j, hi, first=False)
            dump = _dump(lv, pool, dcs, j, lo, hi, sum(lv.jobs[k].V + lv.jobs[k].P for k in out)) \
                if out else None
            if dump is None:
                rejected[j] = reason
                pool.set(j, a, hi)
                continue
            dumped[j] = reason
            d, s, started = dump
            choice = (None, s, None, d, started)
        _, s, _, d, started = choice
        dcs[d].commit(j, s, started)
        bound_at.setdefault(s, []).append(j)
        pool.set(j, a, s)
        plan[j] = (d, s)
    return plan, rejected, dumped


# ─── Phase B ─────────────────────────────────────────────────────────────────

def _repair(lv: _Level, ep: cp.Episode, j: int, s: int, todo: dict, now: dict,
            require: bool = True):
    """The DC a job hidden at its planned step binds to at step s, from the replay's state with
    the remaining planned binds: Phase A's exact check and choice order; with require False (a
    dump) the legal DC where it loses the least value, as _dump chooses. None if there is none."""
    job = lv.jobs[j]
    if require:
        groups, _ = lv.options(j, s)
    else:
        groups = [(None, [(lv.static[d], d, None) for d in lv.legal[j]])]
    for _, members in groups:
        found = []
        for key, d, _latest in members:
            spec, state = lv.specs[d], ep.dcs[d]
            future = {t: list(js) for t, js in todo.get(d, {}).items() if t > s and js}
            base = dict(future)
            if now.get(d):
                base[s] = list(now[d])
            track = unstarted(state)
            for js in base.values():
                track.update(js)
            _, before = run_until_started(state.copy(), s, base, track, (), lv.g)
            met = {k for k, (start, v) in before.items() if on_time(lv.jobs[k], start, spec.vm_mips[v], lv.g)}
            binds = dict(future)
            binds[s] = list(now.get(d, [])) + [j]
            if require:
                ok, started = run_until_started(state.copy(), s, binds, track | {j}, met | {j}, lv.g)
                if ok:
                    start, v = started[j]
                    found.append((start + job.L / spec.vm_mips[v], key, d))
            else:
                _, started = run_until_started(state.copy(), s, binds, track | {j}, (), lv.g)
                cost = job.cost_on(spec, lv.cfg)
                found.append((cost - value_change(lv.jobs, spec, met, started, lv.g), cost, key, d))
        if found:
            return min(found)[-1]
    return None


def _phase_b(lv: _Level, plan: dict) -> dict:
    """Play the plan on the full port; returns the steps, the return and each job's pool span."""
    ep = cp.Episode(lv.params, lv.jobs_json)
    H, K = lv.H, lv.n_slots
    todo = {}                                   # dc -> {step: planned jobs not bound or dropped yet}
    for j, (d, s) in plan.items():
        todo.setdefault(d, {}).setdefault(s, []).append(j)
    done = set()                                # planned jobs bound or dropped
    repaired, dumped, dropped = {}, {}, []
    steps, ret = [], 0.0
    while not ep.terminated:                    # every job resolved, at step H at the latest
        s = ep.step_index
        vis = ep.visible_idx()
        free, backlog = [], []
        for dc in ep.dcs:
            for row in dc.host_rows(ep.clock):
                free.append(row[3])
                backlog.append(row[4])
        actions = [0] * K
        now, hidden = {}, []
        for i, j in enumerate(vis):
            if j not in plan or j in done:
                continue
            d, t = plan[j]
            if t == s:
                actions[i] = d + 1
                now.setdefault(d, []).append(j)
                todo[d][t].remove(j)
                done.add(j)
            elif t < s:
                hidden.append((i, j))
        for i, j in hidden:
            d0, t0 = plan[j]
            todo[d0][t0].remove(j)
            done.add(j)
            d = _repair(lv, ep, j, s, todo, now)
            if d is not None:
                repaired[j] = (d, s)
            else:
                # Left in the pool it could hide planned jobs until its due, as a rejected job
                # would in Phase A: it is dumped.
                d = _repair(lv, ep, j, s, todo, now, require=False)
                if d is None:
                    dropped.append(j)
                    continue
                dumped[j] = (d, s)
            actions[i] = d + 1
            now.setdefault(d, []).append(j)
        info = ep.step(actions)
        ret += info["unshaped_reward"]
        steps.append({"visible": [lv.jobs[j].id for j in vis],
                      "actions": [[i, k] for i, k in enumerate(actions) if k],
                      "free": free, "backlog": backlog,
                      "ledger": {key: info[key] for key in LEDGER_KEYS}})
    spans = []
    for job in lv.jobs:
        a = int(job.a)
        if a > H - 1:
            spans.append((0, -1))
        elif job.idx in ep.bind_step:
            spans.append((a, ep.bind_step[job.idx]))
        else:
            spans.append((a, min(math.floor(job.due), H - 1)))
    lost = [j for j in plan if j not in ep.bound and j not in dropped]
    return {"return": ret, "steps": steps, "spans": spans, "episode": ep,
            "repaired": repaired, "dumped": dumped, "dropped": dropped, "lost": lost}


# ─── Planning ────────────────────────────────────────────────────────────────

def plan_level(params: dict, jobs_json, g: float = R4_GUARD_BAND) -> dict:
    """The R4-CDLS plan of one level (params: the environment's, with datacenters' connect_to
    as indices; jobs_json: env._levels.jobs_json(level)). JSON-serialisable."""
    started = time.perf_counter()
    if not isinstance(jobs_json, str):
        jobs_json = json.dumps(jobs_json, separators=(",", ":"))
    lv = _Level(params, jobs_json, g)
    spans = [(int(job.a), int(job.a)) if job.a <= lv.H - 1 else (0, -1) for job in lv.jobs]
    best, returns = None, []
    for it in range(1, ITERATIONS + 1):
        plan, rejected, dumped = _phase_a(lv, spans)
        replay = _phase_b(lv, plan)
        returns.append(replay["return"])
        if best is None or replay["return"] > best[4]["return"]:
            best = (it, plan, rejected, dumped, replay)
        spans = replay["spans"]
    it, plan, rejected, dumped, replay = best
    jobs, names = lv.jobs, [spec.name for spec in lv.specs]
    ep = replay["episode"]
    reasons = {}
    for reason in rejected.values():
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "version": PLAN_VERSION, "port_version": cp.PORT_VERSION, "guard_band": g,
        "jobs_sha": hashlib.sha256(jobs_json.encode()).hexdigest(), "horizon": lv.H,
        "iteration": it, "iteration_returns": returns, "return": replay["return"],
        "steps": replay["steps"],
        "phase_a": {"binds": {str(jobs[j].id): [names[d], s] for j, (d, s) in sorted(plan.items())},
                    "rejected": {str(jobs[j].id): r for j, r in sorted(rejected.items())},
                    "dumped": {str(jobs[j].id): r for j, r in sorted(dumped.items())}},
        "phase_b": {"binds": {str(jobs[j].id): [names[d], ep.bind_step[j]]
                              for j, (d, _) in sorted(ep.bound.items())},
                    "repaired": sorted(jobs[j].id for j in replay["repaired"]),
                    "dumped": sorted(jobs[j].id for j in replay["dumped"]),
                    "dropped": sorted(jobs[j].id for j in replay["dropped"]),
                    "lost": sorted(jobs[j].id for j in replay["lost"])},
        "stats": {"jobs": len(jobs), "committed": len(plan) - len(dumped), "dumped": len(dumped),
                  "rejected": reasons,
                  "placed": len(ep.bound), "met": sum(1 for met, _ in ep.outcome.values() if met),
                  "repaired": len(replay["repaired"]), "replay_dumped": len(replay["dumped"]),
                  "dropped": len(replay["dropped"]),
                  "lost": len(replay["lost"]),
                  "planning_s": round(time.perf_counter() - started, 3)},
    }


def plan_key(params: dict, jobs_json: str, g: float = R4_GUARD_BAND) -> str:
    """sha256 of what a plan depends on: the port's params, the jobs, PORT_VERSION, g and the
    code of the planner and the port."""
    subset = {k: v for k, v in params.items() if k in PARAM_KEYS or k.startswith(PARAM_PREFIXES)}
    code = hashlib.sha256()
    for module in (__file__, cp.__file__):
        with open(module, "rb") as f:
            code.update(f.read())
    blob = json.dumps([subset, jobs_json, cp.PORT_VERSION, g, PLAN_VERSION, code.hexdigest()],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached_plan(params: dict, jobs_json: str, member: str, level_id: int,
                cache_dir: str = CACHE_DIR, g: float = R4_GUARD_BAND) -> dict:
    """plan_level through the cache <cache_dir>/<member>/<level_id>.json, which holds the plan
    and its plan_key; a plan with another key is replaced."""
    path = os.path.join(cache_dir, member, f"{level_id}.json")
    key = plan_key(params, jobs_json, g)
    try:
        with open(path) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            return cached["plan"]
    except (OSError, ValueError):
        pass
    plan = plan_level(params, jobs_json, g)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump({"key": key, "member": member, "level_id": level_id, "plan": plan}, f,
                  separators=(",", ":"))
    os.replace(tmp, path)
    return plan


# ─── Zero-contention ceiling ─────────────────────────────────────────────────

def zc_ceiling(params: dict, jobs_json) -> tuple:
    """(zc_ceiling, zc_ideal, Z) of one level, both divided by Z = sum of V (spec §5D).

    Per job: -P if it arrives after step H - 1, else the best of -P and, per legal DC, V - c if
    it is on time alone on the empty DC bound at its arrival (a port run), else -P - c.
    zc_ideal replaces the port run with nd + mi / mips <= deadline.
    """
    ep = cp.Episode(params, jobs_json)
    cfg, specs, H = ep.cfg, ep.specs, ep.cfg.horizon
    zc = ideal = 0.0
    for job in ep.jobs:
        best = best_ideal = -job.P
        if job.a <= H - 1:
            s = max(int(job.a), 1)
            for d in cp.legal_dcs(job, specs):
                spec = specs[d]
                cost = job.cost_on(spec, cfg)
                empty = cp.DcState(spec, ep.jobs, cfg)
                empty.step = s
                v = empty.select_vm(job.cores)
                ok, _ = run_until_started(empty, s, {s: [job.idx]}, {job.idx}, {job.idx}, 0.0)
                best = max(best, job.V - cost if ok else -job.P - cost)
                on_time = spec.net_delay + job.L / spec.vm_mips[v] <= job.deadline * cfg.interval
                best_ideal = max(best_ideal, job.V - cost if on_time else -job.P - cost)
        zc += best
        ideal += best_ideal
    Z = ep.offered_value
    return zc / Z, ideal / Z, Z

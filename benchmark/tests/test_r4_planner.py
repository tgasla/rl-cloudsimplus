"""R4-CDLS planning (benchmark/r4_planner.py) on tiny synthetic levels. No JVM needed.

Run: python3 -m pytest benchmark/tests/test_r4_planner.py

The levels use the small topology of test_cloudsim_port.py (cloud 1x32 PE @ 100 MIPS, edge
1x16 @ 60, micro 2x8 @ 40; edge and micro reach each other and the cloud).
"""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cloudsim_port as cp  # noqa: E402
import r4_planner as r4  # noqa: E402
from test_cloudsim_port import EDGE, MICRO, PARAMS, _job  # noqa: E402

TOLERANT, MODERATE, CRITICAL = 0, 1, 2


def random_level(seed, n_jobs=60, last_arrival=25):
    """RING-N-like jobs: mi a multiple of mips_ref, deadline ceil(mi / 40) plus a class slack."""
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_jobs):
        runtime = int(rng.integers(3, 16))
        sens = int(rng.integers(3))
        slack = int(rng.integers(*((6, 25), (2, 9), (0, 3))[sens]))
        jobs.append(_job(i, int(rng.integers(1, last_arrival + 1)), 60 * runtime,
                         int(rng.choice([1, 2, 4, 8, 8, 16])), loc=int(rng.choice([EDGE - 1, MICRO - 1])),
                         sens=sens, deadline=math.ceil(60 * runtime / 40) + slack))
    return sorted(jobs, key=lambda j: (j["submissionDelay"], j["jobId"]))


def replay(params, jobs, plan):
    """Play the plan's actions on a fresh port episode; checks every step against the plan and
    returns the episode."""
    ep = cp.Episode(params, jobs)
    for s, step in enumerate(plan["steps"]):
        assert ep.visible() == step["visible"]
        rows = [row for dc in ep.dcs for row in dc.host_rows(ep.clock)]
        assert [r[3] for r in rows] == step["free"] and [r[4] for r in rows] == step["backlog"]
        actions = [0] * params["max_jobs_waiting"]
        for i, k in step["actions"]:
            actions[i] = k
        info = ep.step(actions)
        assert {key: info[key] for key in r4.LEDGER_KEYS} == step["ledger"]
    assert ep.terminated
    return ep


def ids(ep, jids):
    return {ep.by_id[int(j)] for j in jids}


# ─── Phase A and B ──────────────────────────────────────────────────────────

ROOMY = dict(PARAMS, max_jobs_waiting=32)       # no window pressure: nothing is dumped


@pytest.mark.parametrize("seed", range(4))
def test_every_committed_job_is_met_and_the_replay_is_the_plan(seed):
    jobs = random_level(seed)
    plan = r4.plan_level(ROOMY, jobs)
    ep = replay(ROOMY, jobs, plan)
    committed = set(plan["phase_a"]["binds"])
    assert committed and plan["phase_a"]["dumped"] == {}
    assert plan["phase_b"]["repaired"] == plan["phase_b"]["dumped"] == plan["phase_b"]["lost"] == []
    for jid in committed:
        j = ep.by_id[int(jid)]
        assert ep.outcome[j][0], f"job {jid} was committed but missed its due"
        assert ep.bind_step[j] == plan["phase_a"]["binds"][jid][1]
    assert set(plan["phase_a"]["rejected"]) == {str(job.id) for job in ep.jobs} - committed
    ret = 0.0
    for step in plan["steps"]:
        ret += step["ledger"]["unshaped_reward"]
    assert ret == plan["return"]


def test_a_job_hidden_at_its_planned_step_is_placed_when_it_shows():
    # A window of 8: some rejected jobs push committed ones out of the window (dumping them
    # would lose more), so the replay finds those hidden at their planned step
    jobs = random_level(0)
    plan = r4.plan_level(PARAMS, jobs)
    ep = replay(PARAMS, jobs, plan)
    late = plan["phase_b"]["repaired"] + plan["phase_b"]["dumped"]
    assert late
    for jid in late:
        j = ep.by_id[jid]
        assert ep.bind_step[j] > plan["phase_a"]["binds"][str(jid)][1]
        assert plan["phase_b"]["binds"][str(jid)][1] == ep.bind_step[j]


def test_no_commit_makes_a_met_job_late(monkeypatch):
    history = []
    original = r4._DcPlan.commit

    def audited(self, j, s, started):
        before = set(self.met)
        original(self, j, s, started)
        history.append((j, before, set(self.met)))
        # the tracked starts are those of the DC replayed with its committed binds
        binds = [(k, t) for t, ks in self.binds.items() for k in ks]
        fresh = cp.simulate_dc(cp.DcState(self.spec, self.jobs, self.cfg), binds)
        assert {k: start for k, (start, _) in self.placed.items()} == \
            {k: fresh[k][0] for k in self.placed}

    monkeypatch.setattr(r4._DcPlan, "commit", audited)
    for seed in range(3):
        lv = r4._Level(PARAMS, json.dumps(random_level(seed, n_jobs=45)), r4.R4_GUARD_BAND)
        history.clear()
        plan, _, dumped = r4._phase_a(lv, [(int(job.a), int(job.a)) for job in lv.jobs])
        assert len(history) == len(plan)
        for j, before, after in history:
            if j not in dumped:
                assert before <= after and j in after


@pytest.mark.parametrize("params", [PARAMS, ROOMY], ids=["window8", "window32"])
@pytest.mark.parametrize("seed", range(3))
def test_every_executed_bind_is_visible_and_legal(seed, params):
    jobs = random_level(seed)
    plan = r4.plan_level(params, jobs)
    ep = cp.Episode(params, jobs)
    for step in plan["steps"]:
        for i, k in step["actions"]:
            assert i < len(step["visible"])
            job = ep.jobs[ep.by_id[step["visible"][i]]]
            assert k - 1 in cp.legal_dcs(job, ep.specs)
    assert all(not step["actions"] for step in plan["steps"][:1])     # nothing binds at step 0


def test_rejected_jobs_are_those_without_a_candidate():
    jobs = [
        _job(0, 2, 600, 1, loc=MICRO - 1, sens=CRITICAL, deadline=20),     # cloud: 5 + 6 = 11
        _job(1, 2, 3600, 16, loc=EDGE - 1, sens=TOLERANT, deadline=200),   # 16*60*.005 = 4.8 >= 1.5
        _job(2, 2, 1200, 1, loc=MICRO - 1, sens=MODERATE, deadline=5),     # 15 s at best (cloud 3 + 12)
        _job(3, 3, 600, 16, loc=EDGE - 1, sens=CRITICAL, deadline=11),     # cloud: 6 + 6 = 12 <= 14
        _job(4, 3, 600, 16, loc=EDGE - 1, sens=MODERATE, deadline=11),     # cloud full until 11: edge 4 + 10
        _job(5, 3, 600, 16, loc=EDGE - 1, sens=TOLERANT, deadline=10),     # edge 1.6 >= 1.5, cloud late
    ]
    plan = r4.plan_level(PARAMS, jobs)
    assert plan["phase_a"]["rejected"] == {"1": "cost", "2": "late", "5": "no_candidate"}
    assert plan["phase_a"]["dumped"] == {}
    assert {j: dc for j, (dc, _) in plan["phase_a"]["binds"].items()} == \
        {"0": "cloud", "3": "cloud", "4": "edge"}
    ep = replay(PARAMS, jobs, plan)
    assert all(ep.outcome[ep.by_id[j]][0] for j in (0, 3, 4))


def test_a_rejected_job_that_would_hide_a_committed_one_is_dumped():
    # A window of 2. Job 0 cannot be on time anywhere and is due at 10, ahead of jobs 1 and 2
    # (due 14, 15) which arrive at 5 and are committed there first (denser): left in the pool
    # until its due it would push job 2 out of the window at step 5, so it is bound at once.
    params = dict(PARAMS, max_jobs_waiting=2)
    jobs = [_job(0, 1, 1200, 16, loc=EDGE - 1, sens=MODERATE, deadline=9),
            _job(1, 5, 120, 1, loc=MICRO - 1, sens=CRITICAL, deadline=9),
            _job(2, 5, 120, 1, loc=MICRO - 1, sens=CRITICAL, deadline=10)]
    plan = r4.plan_level(params, jobs)
    assert plan["phase_a"]["dumped"] == {"0": "late"} and plan["phase_a"]["rejected"] == {}
    assert plan["phase_a"]["binds"]["0"][1] == 1
    ep = replay(params, jobs, plan)
    assert [ep.outcome[ep.by_id[j]][0] for j in (0, 1, 2)] == [False, True, True]

    # With room in the window it is simply rejected
    plan = r4.plan_level(dict(PARAMS, max_jobs_waiting=3), jobs)
    assert plan["phase_a"]["rejected"] == {"0": "late"} and plan["phase_a"]["dumped"] == {}


def test_the_queue_check_protects_committed_jobs_that_are_in_the_window():
    # A window of 2 at step 5, where jobs 1, 2 and 3 are bound and hold ranks 0-2: job 3 is
    # already out of the window. Job 0 (due 12, ahead of all three) kept in the pool until step
    # 5 pushes job 2 out (rank 1, the last slot), but neither job 1 nor job 3.
    params = dict(PARAMS, max_jobs_waiting=2)
    jobs = [_job(0, 1, 60, 1, deadline=11), _job(1, 5, 60, 1, deadline=8),
            _job(2, 5, 60, 1, deadline=9), _job(3, 5, 60, 1, deadline=15)]
    lv = r4._Level(params, json.dumps(jobs), r4.R4_GUARD_BAND)
    pool = r4._Pool(lv.krank, lv.H, [(1, 1), (5, 5), (5, 5), (5, 5)])
    bound_at = {5: [1, 2, 3]}
    assert [pool.rank(k, 5) for k in (1, 2, 3)] == [0, 1, 2]
    assert lv.pushes_out(pool, bound_at, 0, 5, first=False) == [2]
    assert lv.pushes_out(pool, bound_at, 0, 4, first=False) == []
    assert not lv.admits(pool, bound_at, 0, 5) and lv.admits(pool, bound_at, 0, 4)


def test_an_uncontended_level_plans_the_zero_contention_ceiling():
    jobs = [_job(i, 1 + 4 * i, 60 * (3 + i % 5), 1 + i % 8, loc=(EDGE, MICRO)[i % 2] - 1,
                 sens=i % 3, deadline=math.ceil(60 * (3 + i % 5) / 40) + 3) for i in range(10)]
    zc, ideal, Z = r4.zc_ceiling(PARAMS, jobs)
    plan = r4.plan_level(PARAMS, jobs)
    assert plan["return"] == pytest.approx(zc, abs=1e-12)
    assert zc == pytest.approx(ideal, abs=1e-12)
    assert Z == cp.Episode(PARAMS, jobs).offered_value


def test_planning_is_deterministic():
    jobs = random_level(7)
    first, second = r4.plan_level(PARAMS, jobs), r4.plan_level(PARAMS, jobs)
    for plan in (first, second):
        del plan["stats"]["planning_s"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def relabel(params, jobs, perm):
    """The same DCs in a new order (position i holds old DC perm[i]), names kept, connect_to
    and job locations renumbered, as PI-S relabels S."""
    new_of_old = {old: new for new, old in enumerate(perm)}
    out = copy.deepcopy(params)
    out["datacenters"] = []
    for old in perm:
        dc = copy.deepcopy(params["datacenters"][old])
        dc["connect_to"] = [new_of_old[c] for c in dc["connect_to"]]
        out["datacenters"].append(dc)
    return out, [dict(job, location=new_of_old[job["location"]]) for job in jobs]


@pytest.mark.parametrize("seed", range(3))
def test_a_relabelled_topology_gets_the_same_plan(seed):
    jobs = random_level(seed)
    params_pi, jobs_pi = relabel(PARAMS, jobs, [2, 0, 1])
    plan, plan_pi = r4.plan_level(PARAMS, jobs), r4.plan_level(params_pi, jobs_pi)
    for phase in ("phase_a", "phase_b"):
        assert plan[phase] == plan_pi[phase]
    assert plan["return"] == plan_pi["return"]
    assert [s["visible"] for s in plan["steps"]] == [s["visible"] for s in plan_pi["steps"]]


# ─── Cache ──────────────────────────────────────────────────────────────────

def test_a_cached_plan_is_reused_until_its_key_changes(tmp_path, monkeypatch):
    jobs = json.dumps(random_level(1, n_jobs=20), separators=(",", ":"))
    plan = r4.cached_plan(PARAMS, jobs, "T", 5, cache_dir=str(tmp_path))
    assert (tmp_path / "T" / "5.json").exists()
    monkeypatch.setattr(r4, "plan_level", lambda *a, **k: pytest.fail("replanned"))
    assert r4.cached_plan(PARAMS, jobs, "T", 5, cache_dir=str(tmp_path)) == plan
    calls = []
    monkeypatch.setattr(r4, "plan_level", lambda *a, **k: calls.append(1) or {"new": True})
    other = dict(PARAMS, sla_penalty_critical=7.0)
    assert r4.cached_plan(other, jobs, "T", 5, cache_dir=str(tmp_path)) == {"new": True}
    assert calls == [1]
    assert r4.plan_key(PARAMS, jobs) != r4.plan_key(dict(PARAMS, grpc_base_port=1), jobs + " ") \
        and r4.plan_key(PARAMS, jobs) == r4.plan_key(dict(PARAMS, grpc_base_port=1), jobs)

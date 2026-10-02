"""The exact CloudSim port (benchmark/cloudsim_port.py). No JVM needed.

Run: python3 -m pytest benchmark/tests/test_cloudsim_port.py

The hand-made scenarios live in SCENARIOS: jobs, binds per step, and the start and finish time of
every job that finishes. `python3 benchmark/tools/diff_port.py --scenarios` plays each of them
through the live gateway jar (sha256 in cloudsim_port.JAR_SHA256) and fails unless the jar's
times are exactly these; the port has to reproduce them bit for bit.

The golden fixtures (data/golden_S_<level>.json) were captured once from the live gateway over
gRPC with benchmark/tools/diff_port.py --members S --levels 2000000,2000001 --capture-fixtures
benchmark/tests/data: member S, test levels 2000000 and 2000001, random legal actions (seed 0).
Each holds the topology and scalar params, the level's jobs, every bind [step, jobId, action],
the jar's per-step unshaped reward and free PEs per DC, and, from the gateway's DEBUG log, every
finished job's [jobId, global VM id, start, finish]. Recapture them (and re-run the differential
test) whenever the jar is rebuilt.
"""
import heapq
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cloudsim_port as cp  # noqa: E402

DATA = os.path.join(HERE, "data")
GOLDEN = [os.path.join(DATA, f"golden_S_{level}.json") for level in (2000000, 2000001)]


# ─── A small topology: cloud 1x32 PE @ 100 MIPS, edge 1x16 @ 60, micro 2x8 @ 40 ──────────

def _dc(name, dc_type, hosts, pes, mips, connect_to=()):
    vm = {"amount": 1, "pes": pes, "pe_mips": mips, "ram": 65536, "size": 1000000, "bw": 10000}
    return {"name": name, "type": dc_type, "amount": 1, "connect_to": list(connect_to),
            "hosts": [{"amount": hosts, "pes": pes, "pe_mips": mips, "ram": 65536,
                       "storage": 1000000, "bw": 10000, "vms": [vm]}]}


PARAMS = {
    "timestep_interval": 1.0, "min_time_between_events": 0.1, "max_episode_length": 60,
    "max_jobs_waiting": 8, "mips_ref": 60, "cloudlet_to_vm_mapping": "most-free-pes",
    "split_large_jobs": False, "reward_shaping": False,
    "sla_value_tolerant": 1.0, "sla_value_moderate": 2.0, "sla_value_critical": 4.0,
    "sla_penalty_tolerant": 0.5, "sla_penalty_moderate": 2.0, "sla_penalty_critical": 6.0,
    "cost_cloud": 0.005, "cost_edge": 0.01, "cost_micro": 0.02,
    "network_delay_cloud": 3.0, "network_delay_edge": 1.0, "network_delay_micro": 0.0,
    "datacenters": [_dc("cloud", "cloud", 1, 32, 100), _dc("edge", "edge", 1, 16, 60, [2, 0]),
                    _dc("micro", "micro", 2, 8, 40, [1, 0])],
}
CLOUD, EDGE, MICRO = 1, 2, 3                 # actions


def _job(i, a, mi, cores, loc=2, sens=0, deadline=100):
    return {"jobId": i, "submissionDelay": a, "mi": mi, "cores": cores, "location": loc,
            "delaySensitivity": sens, "deadline": deadline}


def _run(jobs, binds, params=PARAMS):
    """Play binds {step: [(jobId, action)]} to the end; returns the episode and its step infos."""
    ep = cp.Episode(params, jobs)
    infos = []
    while not ep.terminated:
        slot = {jid: i for i, jid in enumerate(ep.visible())}
        action = [0] * params["max_jobs_waiting"]
        for jid, k in binds.get(ep.step_index, []):
            action[slot[jid]] = k
        infos.append(ep.step(action))
    return ep, infos


def _times(ep):
    return {jid: (r[2], r[3]) for jid, r in ep.results().items()}


def _finished(ep):
    return {jid: sf for jid, sf in _times(ep).items() if sf[1] is not None}


# ─── Scenarios checked against the live jar ─────────────────────────────────
# name: (jobs, binds {step: [(jobId, action)]}, {jobId: (start, finish)} of every finished job)

SCENARIOS = {
    # Bound at step 1 (clock 1.0): the cloud's 3 s network delay, the edge's 1 s, the micro's 0.
    "alone": ([_job(0, 1, 600, 1), _job(1, 1, 600, 1), _job(2, 1, 600, 1)],
              {1: [(0, CLOUD), (1, EDGE), (2, MICRO)]},
              {0: (4.0, 10.0), 1: (2.0, 12.0), 2: (1.0, 16.0)}),
    # The edge VM is full until 12.0; the update that records that finish starts nothing (the
    # first-fit pass runs before the finished job releases its PEs) and schedules the next
    # update at 12.0 + max(0.1, 0.1 + 0.01).
    "wait_0_11": ([_job(0, 1, 600, 16), _job(1, 2, 120, 4)], {1: [(0, EDGE)], 2: [(1, EDGE)]},
                  {0: (2.0, 12.0), 1: (12.11, 14.219999999999999)}),
    # 8 + 8 PEs run on the edge's 16; the 16-PE job 2 queues first, the 8-PE job 3 after it.
    # When job 1 frees 8 PEs at 7.0, job 3 starts at 7.11 and job 2 waits for job 0.
    "first_fit": ([_job(0, 1, 1200, 8), _job(1, 1, 300, 8), _job(2, 2, 60, 16), _job(3, 3, 60, 8)],
                  {1: [(0, EDGE), (1, EDGE)], 2: [(2, EDGE)], 3: [(3, EDGE)]},
                  {0: (2.0, 22.11), 1: (2.0, 7.0), 2: (22.22, 23.33), 3: (7.11, 8.220016666666666)}),
    # Job 1 reaches the edge at 12.0, the time of the update that finishes job 0. The submit runs
    # first, finds the VM full and queues: job 1 starts at the next update, 12.11, not at 12.0.
    "submit_first": ([_job(0, 1, 600, 16), _job(1, 11, 60, 16)], {1: [(0, EDGE)], 11: [(1, EDGE)]},
                     {0: (2.0, 12.0), 1: (12.11, 13.219999999999999)}),
    # Both micro VMs finish their jobs at 16.0, the target of step 15. That update runs at the
    # start of step 16, before the broker sends step 16's binds, so job 2 finds free PEs at 16.0.
    "leftover_first": ([_job(0, 1, 600, 8), _job(1, 1, 600, 8), _job(2, 16, 60, 8)],
                       {1: [(0, MICRO), (1, MICRO)], 16: [(2, MICRO)]},
                       {0: (1.0, 16.0), 1: (1.0, 16.0), 2: (16.0, 17.61)}),
    # Job 1 starts at 16.11 with 15 s to run; the VM reports 15 - 0.11 instead of 15, so the DC's
    # next update is at 31.0, not 31.11, and the finish is recorded at 31.0 + 0.11 + 0.11.
    "decimals": ([_job(0, 1, 600, 8), _job(1, 2, 600, 8), _job(2, 1, 600, 8)],
                 {1: [(0, MICRO), (2, MICRO)], 2: [(1, MICRO)]},
                 {0: (1.0, 16.0), 1: (16.11, 31.22), 2: (1.0, 16.0)}),
    # 4.2 s of work from 4.0: the update at 8.2 credits (long)(100000 * 4.199999999999999) =
    # 419999 of the 420000 units, so the finish is recorded at the next update, 8.2 + 0.11.
    "mi_truncation": ([_job(0, 1, 420, 1, loc=0)], {1: [(0, CLOUD)]}, {0: (4.0, 8.309999999999999)}),
    # Never placed: job 0 (due 4.0) is visible at steps 1-4 and expires when step 4 ends at 5.0.
    "eviction": ([_job(0, 1, 60, 1, deadline=3), _job(1, 2, 60, 1, deadline=30)], {}, {}),
    # Job 0 arrives at 0 and is visible at step 0, but no bind of step 0 finds a VM (the broker
    # lists its VMs during that step): it stays unplaced; job 1, bound at step 1, runs.
    "step0_bind": ([_job(0, 0, 600, 1), _job(1, 1, 600, 1)], {0: [(0, MICRO)], 1: [(1, MICRO)]},
                   {1: (1.0, 16.0)}),
    # Every job finishes inside the last step, as in "alone", but here no keep-alive event is
    # pending past it: CloudSim runs out of events before the step's target and shuts down, and
    # the jar's last observation lists no hosts (P9).
    "shutdown": ([_job(0, 1, 1000, 32, deadline=40), _job(1, 2, 189, 4, deadline=14)],
                 {1: [(0, CLOUD)], 2: [(1, CLOUD)]},
                 {0: (4.0, 14.0), 1: (14.11, 16.11)}),
}


# ─── Scheduling rules (P1-P5) ────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_reproduces_the_jars_times(name):
    jobs, binds, times = SCENARIOS[name]
    ep, _ = _run(jobs, binds)
    assert _finished(ep) == times


def test_a_job_running_alone_finishes_at_start_plus_length_over_mips():
    ep, _ = _run(*SCENARIOS["alone"][:2])
    assert all(start + 600 / mips == fin
               for (start, fin), mips in zip(_times(ep).values(), (100, 60, 40)))


def test_a_waiting_job_starts_0_11_s_after_the_finish_that_frees_its_pes():
    ep, _ = _run(*SCENARIOS["wait_0_11"][:2])
    assert _times(ep)[1][0] == _times(ep)[0][1] + 0.11


def test_first_fit_lets_a_small_later_job_pass_a_large_waiting_one():
    ep, _ = _run(*SCENARIOS["first_fit"][:2])
    times = _times(ep)
    assert times[3][0] < times[2][0]                              # job 3 queued after job 2


def test_a_submit_runs_before_an_update_at_the_same_time():
    ep, _ = _run(*SCENARIOS["submit_first"][:2])
    assert _times(ep)[1][0] > _times(ep)[0][1] == 12.0             # arrived at 12.0, started later


def test_a_zero_delay_submit_runs_after_the_update_left_over_at_the_step_start():
    ep, _ = _run(*SCENARIOS["leftover_first"][:2])
    assert _times(ep)[2][0] == 16.0


def test_no_bind_of_the_first_step_finds_a_vm():
    ep, infos = _run(*SCENARIOS["step0_bind"][:2])
    assert [info["jobs_placed"] for info in infos[:2]] == [0, 1]
    assert ep.results()[0] == (None, None, None, None, False)       # expired unplaced
    with pytest.raises(ValueError, match="places nothing"):
        cp.simulate_dc(cp.DcState(ep.specs[MICRO - 1], ep.jobs, ep.cfg), [(0, 0)])


def test_an_update_inside_the_0_1_s_gate_does_nothing():
    ep = cp.Episode(PARAMS, [_job(0, 1, 600, 16)])
    edge = ep.dcs[EDGE - 1]
    edge.submit(0, 0, 5.0)
    edge.run_step(5.0, 6.0)                  # the job starts at 5.0, its update is due at 15.0
    edge.last_process = 5.0
    cle = edge.exec[0][0]
    heapq.heappush(edge.heap, (5.05, cp.TAG_UPDATE, edge.serial, -1, -1))   # 0.05 s after it
    before = (cle.done, cle.last, list(edge.heap))
    edge.run_step(5.0, 6.0)
    assert edge.last_process == 5.0 and (cle.done, cle.last) == before[:2]
    assert edge.heap == [e for e in before[2] if e[0] != 5.05]     # nothing scheduled
    heapq.heappush(edge.heap, (5.1, cp.TAG_UPDATE, edge.serial, -1, -1))    # 0.1 s after: it runs
    edge.run_step(5.0, 6.0)
    assert edge.last_process == 5.1 and cle.done == int(60000.0 * (5.1 - 5.0))


def test_the_decimals_quirk_pulls_the_next_update_to_an_integer_time():
    # VmAbstract.cloudletsProcessing: job 1 starts at 16.11 with 15 s to run, and the DC's next
    # update is due at 31.0 (15 - frac(16.11) after it), not at 31.11.
    jobs, binds, _ = SCENARIOS["decimals"]
    ep = cp.Episode(PARAMS, jobs)
    for s in range(17):
        slot = {jid: i for i, jid in enumerate(ep.visible())}
        action = [0] * 8
        for jid, k in binds.get(s, []):
            action[slot[jid]] = k
        ep.step(action)
    micro = ep.dcs[MICRO - 1]
    assert micro.start[1] == 16.11
    assert [t for t, *_ in micro.heap] == [31.0]


def test_mi_truncation_records_a_finish_one_update_late():
    assert int(100000.0 * ((4.0 + 420000 / 100000.0) - 4.0)) == 419999
    ep, _ = _run(*SCENARIOS["mi_truncation"][:2])
    start, finish = _times(ep)[0]
    assert finish == start + 420000 / 100000.0 + 0.11              # one update after 8.2


# ─── P8: the VM selector ─────────────────────────────────────────────────────

def _micro_state(host_pes=(8, 8)):
    params = dict(PARAMS, datacenters=[
        {"name": "m", "type": "micro", "amount": 1, "connect_to": [],
         "hosts": [{"amount": 1, "pes": p, "pe_mips": 40, "vms": [
             {"amount": 1, "pes": p, "pe_mips": 40, "size": 1000000}]} for p in host_pes]}])
    jobs = [_job(i, 1, 60, c, loc=0) for i, c in enumerate((4, 8, 8, 2, 6))]
    ep = cp.Episode(params, jobs)
    return ep.dcs[0]


def test_vm_selector_takes_the_first_strict_maximum_of_expected_free_pes():
    dc = _micro_state()
    assert dc.select_vm(4) == 0                                   # a tie goes to the first VM
    dc.exec[0].append(cp.Cle(0, 4, 60000))                        # VM 0 runs 4 PEs
    assert dc.select_vm(4) == 1
    dc.inflight[1] = (1, 7.0)                                     # 8 PEs in flight to VM 1
    assert dc.select_vm(4) == 0                                   # 4 free beats 0
    assert dc.select_vm(4, {0: 4}) == 0                           # 0 vs 0: the first again
    assert dc.select_vm(4, {0: 6}) == 1                           # -2 vs 0
    dc.wait[1].append(cp.Cle(2, 8, 60000))                        # queued PEs count too
    assert dc.select_vm(4, {0: 6}) == 0                           # -2 vs -8
    assert dc.select_vm(9) is None                                # no VM can hold 9 PEs


def test_vm_selector_skips_vms_too_small_for_the_job():
    dc = _micro_state(host_pes=(4, 8))
    dc.exec[1].append(cp.Cle(0, 6, 60000))                        # VM 1: 2 free, VM 0: 4 free
    assert dc.select_vm(2) == 0
    assert dc.select_vm(6) == 1                                   # VM 0 is too small


# ─── P7, P6: visibility and the ledger ───────────────────────────────────────

def test_an_unplaced_job_is_evicted_once_the_clock_passes_its_due_time():
    # due = 1 + 3 = 4.0: still visible at step 4 (clock 4.0 is not past 4.0), resolved as expired
    # when step 4 ends at clock 5.0, gone at step 5.
    ep = cp.Episode(PARAMS, SCENARIOS["eviction"][0])
    seen, expired = [], []
    while not ep.terminated:
        seen.append(ep.visible())
        expired.append(ep.step([0] * 8)["jobs_expired_unplaced"])
    assert seen[:6] == [[], [0], [0, 1], [0, 1], [0, 1], [1]]
    assert [s for s, n in enumerate(expired) if n] == [4, 32]
    assert ep.results()[0][4] is False


def _visible_from_scratch(ep, bound_before, clock, target):
    """P7 recomputed from the definitions: arrived, not submitted, not past due."""
    jobs = [j for j in ep.jobs if j.a < target and j.idx not in bound_before
            and not (clock > j.due)]
    jobs.sort(key=lambda j: (j.due, j.a, j.id))
    return [j.id for j in jobs[: ep.cfg.max_jobs_waiting]]


@pytest.mark.parametrize("path", GOLDEN[:1])
def test_visible_window_equals_a_direct_recomputation(path):
    fx = _fixture(path)
    ep = cp.Episode(fx["params"], fx["jobs"])
    by_step = _binds_by_step(fx)
    while not ep.terminated:
        bound_before = set(ep.bound)
        assert ep.visible() == _visible_from_scratch(ep, bound_before, ep.clock, ep.target())
        ep.step(_slot_actions(ep, by_step.get(ep.step_index, [])))


# The busy micro DC of the gateway's SlaRewardTest (env_b_params.json, 3 hosts x 6 PEs @ 60 MIPS,
# every job to it): job 7's execution ends 0.05 ms before its due, CloudSim records it 0.110017 s
# after. Found with this port; the jar's JUnit asserts the same outcome.
ENV_B = os.path.join(os.path.dirname(os.path.dirname(HERE)), "domain", "job-placement",
                     "cloudsimplus-gateway", "src", "test", "resources", "env_b_params.json")
BUSY_MICRO = [(0, 1, 1452, 3, 100), (4, 2, 239, 3, 100), (6, 4, 245, 3, 100), (7, 5, 233, 3, 5),
              (10, 3, 191, 3, 100), (12, 2, 99, 3, 100), (16, 1, 298, 1, 100), (17, 4, 113, 2, 100),
              (19, 2, 51, 2, 100)]


def _env_b_episode(jobs):
    """Every visible job to micro_dc_ucd (action 3), to the end."""
    with open(ENV_B) as f:
        params = dict(json.load(f), split_large_jobs=False)
    ep = cp.Episode(params, [_job(i, a, mi, cores, deadline=dl) for i, a, mi, cores, dl in jobs])
    while not ep.terminated:
        ep.step([3] * len(ep.visible()) + [0] * (params["max_jobs_waiting"] - len(ep.visible())))
    return ep


def test_on_time_is_judged_from_the_execution_not_the_recorded_finish():
    ep = _env_b_episode(BUSY_MICRO)
    j = ep.by_id[7]
    d, v, start, finish, met = ep.results()[7]
    assert start + ep.jobs[j].L / ep.specs[d].vm_mips[v] == 9.99995 and ep.jobs[j].due == 10.0
    assert finish == 10.110016666666665                           # past due + 0.11
    assert met and ep.on_time(j)
    assert all(m for *_, m in ep.results().values())


def test_a_job_whose_execution_ends_after_its_due_is_violated_even_if_recorded_within_0_11():
    ep = _env_b_episode([(0, 1, 603, 1, 10)])                      # ends 11.05, due 11
    assert ep.results()[0][3] == 11.05 and ep.results()[0][4] is False


# ─── Golden traces from the live jar ─────────────────────────────────────────

def _fixture(path):
    with open(path) as f:
        fx = json.load(f)
    fx["jobs"] = [dict(zip(fx["job_fields"], row)) for row in fx["jobs"]]
    return fx


def _binds_by_step(fx):
    out = {}
    for step, jid, k in fx["binds"]:
        out.setdefault(step, []).append((jid, k))
    return out


def _slot_actions(ep, binds):
    slot = {jid: i for i, jid in enumerate(ep.visible())}
    action = [0] * ep.cfg.max_jobs_waiting
    for jid, k in binds:
        action[slot[jid]] = k
    return action


@pytest.mark.parametrize("path", GOLDEN)
def test_golden_trace_is_reproduced_exactly(path):
    fx = _fixture(path)
    assert fx["captured_with"]["jar_sha256"] == cp.JAR_SHA256, "fixture from another jar: recapture it"
    ep = cp.Episode(fx["params"], fx["jobs"])
    by_step = _binds_by_step(fx)
    rewards, free, ret = [], [], 0.0
    while not ep.terminated:
        info = ep.step(_slot_actions(ep, by_step.get(ep.step_index, [])))
        rewards.append(info["unshaped_reward"])
        free.append([sum(row[3] for row in dc.host_rows(ep.clock)) for dc in ep.dcs])
        ret += info["unshaped_reward"]
    assert rewards == fx["java_unshaped_reward"]
    assert free == fx["java_free_pes_by_dc"]
    assert ret == fx["java_return"]
    vm_base = [sum(s.n_vms for s in ep.specs[:d]) for d in range(len(ep.specs))]
    port = sorted([jid, vm_base[d] + v, start, fin] for jid, (d, v, start, fin, _) in ep.results().items()
                  if fin is not None)
    assert port == fx["java_cloudlets"]


@pytest.mark.parametrize("path", GOLDEN[:1])
def test_simulate_dc_reproduces_the_episode_per_dc(path):
    fx = _fixture(path)
    ep = cp.Episode(fx["params"], fx["jobs"])
    by_step = _binds_by_step(fx)
    snaps = {}
    while not ep.terminated:
        if ep.step_index == 60:
            snaps = {d: dc.copy() for d, dc in enumerate(ep.dcs)}
        ep.step(_slot_actions(ep, by_step.get(ep.step_index, [])))
    for d, dc in enumerate(ep.dcs):
        binds = [(j, ep.bind_step[j]) for j, (dd, _) in ep.bound.items() if dd == d]
        want = {j: (dc.start.get(j), dc.finish.get(j)) for j in dc.vm_of}
        fresh = cp.DcState(ep.specs[d], ep.jobs, ep.cfg)
        record = {}
        got = cp.simulate_dc(fresh, binds, record=record)
        # the drain stops at the horizon's resolution, simulate_dc runs every job to its end
        assert {j: sf for j, sf in got.items() if want[j][1] is not None} == \
            {j: sf for j, sf in want.items() if sf[1] is not None}
        late = [(j, s) for j, s in binds if s >= 60]
        assert cp.simulate_dc(fresh, late, from_snapshot=snaps[d]) == got
        assert cp.simulate_dc(fresh, binds, reference=record) == got


@pytest.mark.parametrize("path", GOLDEN[:1])
def test_an_episode_copy_is_an_independent_branch(path):
    fx = _fixture(path)
    by_step = _binds_by_step(fx)

    def play(ep, until=None):
        infos = []
        while not ep.terminated and (until is None or ep.step_index < until):
            visible = set(ep.visible())               # a branch may not see every recorded bind
            binds = [(jid, k) for jid, k in by_step.get(ep.step_index, []) if jid in visible]
            infos.append(ep.step(_slot_actions(ep, binds)))
        return infos

    ep = cp.Episode(fx["params"], fx["jobs"])
    play(ep, until=60)
    branch = ep.copy()
    branch.step([0] * ep.cfg.max_jobs_waiting)         # the branch defers step 60's jobs
    play(branch)
    rest = play(ep)                                     # the original is untouched by the branch
    assert [info["unshaped_reward"] for info in rest] == fx["java_unshaped_reward"][60:]
    assert branch.results() != ep.results()
    replay = cp.Episode(fx["params"], fx["jobs"])
    play(replay)
    assert replay.results() == ep.results()


def test_simulate_dc_early_exit_takes_over_the_reference_run():
    jobs = [_job(0, 1, 600, 16), _job(1, 2, 120, 4), _job(2, 30, 60, 4)]
    ep = cp.Episode(PARAMS, jobs)
    edge = cp.DcState(ep.specs[EDGE - 1], ep.jobs, ep.cfg)
    record = {}
    full = cp.simulate_dc(edge, [(0, 1), (1, 2), (2, 30)], record=record)
    # the same binds: the state after the last bind matches the recorded one at once
    assert cp.simulate_dc(edge, [(0, 1), (1, 2), (2, 30)], reference=record) == full
    # without job 2 the state differs from step 30 on, so nothing is taken over from there
    assert cp.simulate_dc(edge, [(0, 1), (1, 2)], reference=record) == \
        {j: sf for j, sf in full.items() if j != 2}
    # a run that took over hands the later keys on, so it can be the next run's reference
    chained = {}
    assert cp.simulate_dc(edge, [(0, 1), (1, 2), (2, 30)], reference=record, record=chained) == full
    assert chained["keys"] == record["keys"]
    assert cp.simulate_dc(edge, [(0, 1), (1, 2), (2, 30)], reference=chained) == full
    with pytest.raises(ValueError):                  # a reference cut at another clock
        cp.simulate_dc(edge, [(0, 1)], until=20.0, reference=record)
    cut = {}
    early = cp.simulate_dc(edge, [(0, 1), (1, 2)], until=13.0, record=cut)
    assert early == {0: (2.0, 12.0), 1: (12.11, None)}
    assert cp.simulate_dc(edge, [(0, 1), (1, 2)], until=13.0, reference=cut) == early


# ─── Plumbing ────────────────────────────────────────────────────────────────

def test_java_dsum_matches_javas_compensated_doublestream_sum():
    # Arrays and DoubleStream.sum() results from JDK 25; a plain left-to-right sum differs.
    cases = [
        ([65.31657230959125, 86.63769447264403, 7.767088930498301, 88.86677716147179,
          35.309848117549045], 283.8979809917544),
        ([0.1, 0.2, 0.3], 0.6),
    ]
    for values, java in cases:
        assert cp.java_dsum(values) == java
        assert sum(values) != java


def test_port_version_pins_the_validated_jar():
    assert cp.JAR_SHA256 in cp.PORT_VERSION
    if not os.path.exists(cp.JAR_PATH):
        pytest.skip("gateway jar not built")
    assert cp.jar_sha256() == cp.JAR_SHA256, (
        "the gateway jar changed: re-run benchmark/tools/diff_port.py, recapture the golden "
        "fixtures, then update cloudsim_port.JAR_SHA256")


def test_unsupported_settings_are_refused():
    with pytest.raises(ValueError):
        cp.Episode(dict(PARAMS, reward_shaping=True), [])
    with pytest.raises(ValueError):
        cp.Episode(dict(PARAMS, cloudlet_to_dc_mapping="earliest-shortest-to-most-free-dc"), [])
    two_vms = json.loads(json.dumps(PARAMS))
    two_vms["datacenters"][0]["hosts"][0]["vms"][0].update(amount=2, pes=16)
    with pytest.raises(ValueError):
        cp.Episode(two_vms, [])
    with pytest.raises(ValueError):
        cp.Episode(PARAMS, [_job(0, 60, 60, 1)])                  # arrives at the horizon

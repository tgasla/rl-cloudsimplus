"""Exact Python replica of how the job-placement gateway schedules, runs and scores jobs.

The planning model of the R4 reference policy and the oracle of the differential test. It
replays, float for float, what the gateway jar does with a sequence of binds: CloudSim Plus
9.0.0-SNAPSHOT (event queue, DatacenterSimple, VmAbstract, CloudletSchedulerAbstract /
SpaceShared, CloudletExecution) as driven by daislab.cspg (OptimizedCloudletScheduler,
CloudSimProxy, WrappedSimulation, SlaLedger). Each rule names the Java it comes from; P1-P9 are
the rules of the reference-policy spec, each re-derived from those sources.

P1  Events run in (time, tag, serial) order: at one time a CLOUDLET_SUBMIT (tag 16) runs before a
    VM_UPDATE_CLOUDLET_PROCESSING (tag 41). A submit starts or queues its job and schedules an
    update at the job's finish (DatacenterSimple.submitCloudletToVm); it updates no running job.
    CloudSim moves each batch of same-time events to its deferred queue and the entities run
    them at the start of the next CloudSim.processEvents call, so a batch due exactly at a step's
    target time runs at the start of the next step, after that step's binds (see P2).
P2  Step driver (CloudSimProxyBase.runOneTimestep, CloudSimProxy.tryToSubmitJobs): a step starts
    at clock c (min_time_between_events before the first step, then each step's target) and runs
    to T = c + interval (T = interval on the first step). Bound jobs are submitted in
    (arrival, id) order and reach their DC at c + max(a - c, 0) + network_delay. At time c a DC
    first runs the batch left over from the previous step, then the jobs submitted this step with
    zero network delay (the broker sends them from a CLOUDLET_CREATION event at c). Every event
    before T runs within the step.
P3  Submit (CloudletSchedulerAbstract.cloudletSubmitInternal): the job runs at once if its VM has
    the PEs free, with an update at t + L / mips; otherwise it joins the end of the waiting list.
P4  DC update (DatacenterSimple.updateCloudletProcessing): skipped when t < lastProcess + 0.1
    (and t >= 0.111). Otherwise every VM is updated in host order, the next update is scheduled
    at t + max(smallest positive VM delay, 0.1 + 0.01) (HostAbstract ignores a VM delay <= 0),
    and lastProcess = t.
P5  VM update (CloudletSchedulerAbstract.updateProcessing, OptimizedCloudletScheduler,
    VmAbstract.cloudletsProcessing): a running job advances (long)((long) mips * (t - last)) and
    finishes at the update where its work reaches L; its estimate is max(remaining / mips, 0.1).
    The first-fit pass moves waiting jobs to exec BEFORE the finished jobs release their PEs.
    0.1 if the waiting list changed and nothing else is due; then the "decimals" quirk returns
    next - frac(t) when that is not negative.
P6  Ledger (SlaLedger, WrappedSimulation.step): after each step, in job-list order, a job is on
    time iff its execution ends by its due, start + L / vm_mips <= due + 1e-9 (executionEnd; the
    recorded finish lags it by up to ~0.12 s). A finished job is met iff on time; an unfinished
    one is violated once the clock is past its due, unless it is running and on time (it then
    waits for its recorded finish), and evicted if it was never placed; c = kappa * cores * L /
    mips_ref. At step H every unplaced job expires and the simulation runs on, one interval at
    a time, until every job resolved.
P7  Visible jobs (CloudSimProxy.getVisibleJobs): neither submitted nor evicted, a < T, sorted by
    (due, arrival, id), the first max_jobs_waiting.
P8  VM selector (WrappedSimulation.getMostFreeVmOfDcForCloudlet): among the DC's VMs with
    pes >= cores, the first strict maximum of pes - sum(pes of its exec and waiting jobs) -
    sum(pes of jobs in flight to it) - pes bound to it earlier in the step. It walks the
    broker's VM list, which is empty during the first step (the VM creation acks are due at
    min_time_between_events and run in that step's clock advance), so every bind of step 0 is
    skipped and its job stays unplaced.
P9  Observation (WrappedSimulation.getInfraObsPerHost, CloudSimProxy.getJobsWaitingObservation):
    reproduced exactly, including Java's compensated DoubleStream.sum in the backlog, except the
    last observation of an episode in which every job finished. CloudSim shuts down once it runs
    out of events (CloudSimPlus.runFor, CloudSim.finish), and CloudInformationService.shutdown
    clears the datacenter list the host rows are read from: if that happens before the last
    step's target, the jar lists no hosts. Whether it does depends on the keep-alive events of
    CloudSimProxyBase.ensureAllJobsCompleteBeforeSimulationEnds (whenever the event queue is down
    to one event while a job is unfinished, an empty event one interval later), which the port
    does not model: it always gives the rows. That observation is never acted on, and episodes
    terminate, never truncate.

Datacenters never interact: a DC's schedule depends only on which jobs are bound to it and at
which step. The only coupling is the visible window, which depends only on the binds and time.

Units are the simulator's: work is counted in 1/MI_RESOLUTION MI (SimulationSettings), so a job's
length is L = mi * 1000 and a PE runs pe_mips * 1000 of them per second.

Validated for topologies with one VM per host, as large as the host (JobPlacementEnv enforces
that), against the jar whose sha256 PORT_VERSION carries: benchmark/tools/diff_port.py plays the
same actions through the port and a live gateway and compares every step (slots, host rows,
ledger) and every job's VM, start and finish; it found no difference at all, in any quantity, on
S, GAM-lo and LOCK test levels under random, deferring, origin-only and cloud-only policies, on
C1-N19, C2-N7, PI-S and GAM-hi test levels under random and deferring policies, nor on the
hand-made scenarios of benchmark/tests/test_cloudsim_port.py (benchmark/results/diff_port*.json).
Re-run it after any jar rebuild.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sys

# The jar the port was validated against (benchmark/tools/diff_port.py and the golden fixtures in
# benchmark/tests/data). After a jar rebuild, re-run both and update this.
JAR_SHA256 = "9cf72c79723e866c06053526262ad2340a82b0a356646358baae00ddc8d3a69d"
PORT_VERSION = f"cloudsim_port/3 jar-sha256:{JAR_SHA256}"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
JAR_PATH = os.path.join(REPO, "domain", "job-placement", "cloudsimplus-gateway", "build", "libs",
                        "cloudsimplus-gateway-0.1.0.jar")

DOUBLE_MAX = sys.float_info.max             # Double.MAX_VALUE
LONG_MIN = -(2 ** 63)                       # Long.MIN_VALUE
TAG_SUBMIT = 16                             # CloudSimTag.CLOUDLET_SUBMIT
TAG_UPDATE = 41                             # CloudSimTag.VM_UPDATE_CLOUDLET_PROCESSING
MI_RESOLUTION = 1000.0                      # SimulationSettings.MI_RESOLUTION
CLOUDSIM_UPDATE_MARGIN = 0.01               # DatacenterSimple: next update >= min_time_between_events + this
ON_TIME_EPSILON = 1e-9                      # SlaLedger.ON_TIME_EPSILON
FIRST_UPDATE_WINDOW = 0.111                 # DatacenterSimple.isTimeToUpdateCloudletsProcessing
MAX_CLOCK_ITERATIONS = 1000                 # CloudSimProxyBase.proceedClockTo gives up after these
DC_TYPE_IDS = {"cloud": 1, "edge": 2, "micro": 3}
SENSITIVITIES = ("tolerant", "moderate", "critical")
JOB_WIRE_FEATURES = 7                       # CloudSimProxy.JOB_OBS_FEATURES
HOST_OBS_FEATURES = 5                       # WrappedSimulation.HOST_OBS_FEATURES
CLOUDLET_FILE_SIZE = 1500                   # DataCloudTags.DEFAULT_MTU; a VM must hold it


class PortError(RuntimeError):
    """The episode left the regime the port reproduces exactly."""


def jar_sha256(path: str = JAR_PATH) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def java_dsum(values) -> float:
    """DoubleStream.sum() of JDK 18+ (Collectors.sumWithCompensation, computeFinalSum)."""
    s0 = s1 = simple = 0.0
    for v in values:
        tmp = v - s1
        velvel = s0 + tmp
        s1 = (velvel - s0) - tmp
        s0 = velvel
        simple += v
    total = s0 - s1
    if math.isnan(total) and math.isinf(simple):
        return simple
    return total


def _scaled_mips(pe_mips) -> float:
    """CloudSimProxy.scaledMips, (long) (pe_mips * MI_RESOLUTION), as a Vm's double MIPS."""
    return float(int(float(pe_mips) * MI_RESOLUTION))


# ─── Static model ────────────────────────────────────────────────────────────

class Settings:
    """The SimulationSettings values the port needs, computed as Java computes them."""

    def __init__(self, params: dict, horizon: int | None = None):
        self.interval = float(params["timestep_interval"])
        self.min_t = float(params["min_time_between_events"])
        self.horizon = int(params["max_episode_length"] if horizon is None else horizon)
        self.max_jobs_waiting = int(params["max_jobs_waiting"])
        self.mips_ref = float(params["mips_ref"]) * MI_RESOLUTION
        self.sla_value = [float(params[f"sla_value_{s}"]) for s in SENSITIVITIES]
        self.sla_penalty = [float(params[f"sla_penalty_{s}"]) for s in SENSITIVITIES]
        self.cost = {t: float(params[f"cost_{t}"]) for t in DC_TYPE_IDS}
        # SimulationSettings.networkDelay: timesteps * interval
        self.net_delay = {t: float(params[f"network_delay_{t}"]) * self.interval for t in DC_TYPE_IDS}
        self.update_floor = self.min_t + CLOUDSIM_UPDATE_MARGIN  # DatacenterSimple.updateHostsProcessing
        if params.get("cloudlet_to_dc_mapping", "rl") != "rl":
            raise ValueError("the port replays actions (cloudlet_to_dc_mapping: rl); with a rule-based "
                             "mapping the jar ignores them")
        if params.get("cloudlet_to_vm_mapping", "most-free-pes") != "most-free-pes":
            raise ValueError("the port models the most-free-pes VM selector only")
        if params.get("split_large_jobs"):
            raise ValueError("the port models unsplit jobs only (split_large_jobs: false)")
        if int(params.get("drift_at_step", -1) or -1) > 0:
            raise ValueError("the port does not model drift injection")
        if params.get("reward_shaping"):
            raise ValueError("the port reproduces the unshaped reward only (reward_shaping: false)")

    def step_clock(self, s: int) -> float:
        """The clock at the start of step s: min_time_between_events, then each step's target,
        by repeated addition as the simulator advances."""
        if s == 0:
            return self.min_t
        c = self.interval
        for _ in range(s - 1):
            c += self.interval
        return c

    def first_visible_step(self, a: float) -> int:
        """The first step whose target time exceeds arrival a (the job's first visible step)."""
        s, target = 0, self.interval
        while not a < target:
            s += 1
            target += self.interval
        return s


class DcSpec:
    """One datacenter: its VMs in the broker's order (host order), one per host."""

    def __init__(self, index: int, dc: dict, cfg: Settings):
        self.index = index
        self.name = dc["name"]
        self.type = dc["type"]
        self.type_id = DC_TYPE_IDS[self.type]
        self.net_delay = cfg.net_delay[self.type]
        self.kappa = cfg.cost[self.type]
        if int(dc.get("amount", 1)) != 1:
            raise ValueError(f"{self.name}: datacenter amount must be 1")
        self.vm_pes, self.vm_mips = [], []
        for host in dc["hosts"]:
            vms = host["vms"] if isinstance(host["vms"], list) else [host["vms"]]
            if (len(vms) != 1 or int(vms[0].get("amount", 1)) != 1 or vms[0]["pes"] != host["pes"]
                    or vms[0]["pe_mips"] != host["pe_mips"] or vms[0]["size"] < CLOUDLET_FILE_SIZE):
                raise ValueError(f"{self.name}: the port is validated for one VM per host, as large "
                                 f"as the host, got {vms}")
            for _ in range(int(host.get("amount", 1))):
                self.vm_pes.append(int(vms[0]["pes"]))
                self.vm_mips.append(_scaled_mips(vms[0]["pe_mips"]))
        self.n_vms = len(self.vm_pes)
        self.capacity = sum(self.vm_pes)
        self.max_vm_pes = max(self.vm_pes)
        self.connect_to = [int(c) for c in dc.get("connect_to", [])]


class Job:
    __slots__ = ("idx", "id", "a", "mi", "L", "cores", "loc", "sens", "deadline", "due", "V", "P", "r")

    def __init__(self, idx: int, spec: dict, cfg: Settings):
        self.idx = idx
        self.id = int(spec["jobId"])
        self.a = float(int(spec["submissionDelay"]))             # CloudletDescriptor: long
        self.mi = int(spec["mi"])
        self.L = int(self.mi * MI_RESOLUTION)                    # CloudletDescriptorWithLocation
        self.cores = int(spec["cores"])
        self.loc = int(spec["location"])
        self.sens = int(spec["delaySensitivity"])
        self.deadline = int(spec["deadline"])
        self.due = self.a + self.deadline * cfg.interval         # CloudSimProxy.getDueTime
        self.V = cfg.sla_value[self.sens]
        self.P = cfg.sla_penalty[self.sens]
        # CloudSimProxy.getJobsWaitingObservation: ceil(length / (mips_ref * interval))
        self.r = int(math.ceil(self.L / (cfg.mips_ref * cfg.interval)))

    def cost_on(self, dc: DcSpec, cfg: Settings) -> float:
        """WrappedSimulation.bind: costPerRefCoreSecond * pes * length / mips_ref."""
        return dc.kappa * self.cores * self.L / cfg.mips_ref


def parse_jobs(jobs, cfg: Settings) -> list[Job]:
    """Jobs of a jobs_json payload (string or list), in the simulator's list order."""
    if isinstance(jobs, str):
        jobs = json.loads(jobs)
    return [Job(i, j, cfg) for i, j in enumerate(jobs)]


def legal_dcs(job: Job, specs: list[DcSpec]) -> list[int]:
    """DC indices the action mask allows for a job: reach (origin plus connect_to; every DC when
    the origin connects nowhere) and a VM that can hold it (JobPlacementEnv.action_masks)."""
    origin = specs[job.loc]
    reach = range(len(specs)) if not origin.connect_to else [job.loc] + origin.connect_to
    return [d for d in reach if specs[d].max_vm_pes >= job.cores]


# ─── One datacenter ──────────────────────────────────────────────────────────

class Cle:
    """A CloudletExecution and the cloudlet fields the scheduler reads."""
    __slots__ = ("j", "pes", "L", "done", "fin", "last")

    def __init__(self, j: int, pes: int, L: int):
        self.j = j            # job index
        self.pes = pes
        self.L = L
        self.done = 0         # CloudletExecution.partialFinishedMI (not capped)
        self.fin = 0          # Cloudlet.finishedLengthSoFar (capped at L)
        self.last = -1.0      # CloudletExecution.lastProcessingTime

    def copy(self) -> "Cle":
        c = Cle.__new__(Cle)
        c.j, c.pes, c.L, c.done, c.fin, c.last = self.j, self.pes, self.L, self.done, self.fin, self.last
        return c


class DcState:
    """The mutable state of one datacenter at the start of step `step`, before its binds: the VM
    schedulers, the pending events, the jobs in flight, and the start/finish times so far.

    copy() (alias snapshot()) branches it; copies share the spec and the job table (read-only).
    """

    def __init__(self, spec: DcSpec, jobs: list[Job], cfg: Settings):
        self.spec, self.jobs, self.cfg = spec, jobs, cfg
        n = spec.n_vms
        self.step = 0
        self.alloc = [0] * n                  # Processor allocated PEs per VM
        self.exec = [[] for _ in range(n)]    # cloudletExecList per VM
        self.wait = [[] for _ in range(n)]    # cloudletWaitingList per VM
        self.heap = []                        # pending events (time, tag, serial, job, vm)
        self.serial = 0
        self.last_process = 0.0               # DatacenterSimple.lastProcessTime
        self.inflight = {}                    # job -> (vm, arrival), in submission order
        self.vm_of = {}                       # job -> vm, every job submitted here
        self.start = {}                       # job -> start time
        self.finish = {}                      # job -> finish time
        self.times = set()                    # event times run in the current step
        self.new_finished = []                # jobs that finished in the current step

    def copy(self) -> "DcState":
        c = DcState.__new__(DcState)
        c.spec, c.jobs, c.cfg = self.spec, self.jobs, self.cfg
        c.step = self.step
        c.alloc = list(self.alloc)
        c.exec = [[x.copy() for x in lst] for lst in self.exec]
        c.wait = [[x.copy() for x in lst] for lst in self.wait]
        c.heap = list(self.heap)
        c.serial = self.serial
        c.last_process = self.last_process
        c.inflight = dict(self.inflight)
        c.vm_of = dict(self.vm_of)
        c.start = dict(self.start)
        c.finish = dict(self.finish)
        c.times = set()
        c.new_finished = []
        return c

    snapshot = copy

    def key(self) -> tuple:
        """Everything that decides this DC's future, to compare two states at a step boundary."""
        return (
            self.last_process,
            tuple(self.alloc),
            tuple(tuple((x.j, x.done, x.fin, x.last) for x in lst) for lst in self.exec),
            tuple(tuple(x.j for x in lst) for lst in self.wait),
            tuple((t, tag, j, v) for t, tag, _, j, v in sorted(self.heap)),
            tuple(self.inflight.items()),
        )

    def busy(self) -> bool:
        return bool(self.heap or self.inflight or any(self.exec) or any(self.wait))

    # ── P8 ──
    def select_vm(self, cores: int, bound_pes: dict | None = None) -> int | None:
        """The VM a job of `cores` PEs bound now goes to; bound_pes: PEs bound to each VM of this
        DC earlier in the step. None if no VM can ever hold it."""
        pes, jobs = self.spec.vm_pes, self.jobs
        inflight_pes = [0] * len(pes)
        for j, (v, _) in self.inflight.items():
            inflight_pes[v] += jobs[j].cores
        best, best_free = None, LONG_MIN
        for v in range(len(pes)):
            free = (pes[v] - sum(x.pes for x in self.exec[v]) - sum(x.pes for x in self.wait[v])
                    - inflight_pes[v] - (bound_pes.get(v, 0) if bound_pes else 0))
            if pes[v] >= cores and free > best_free:
                best, best_free = v, free
        return best

    def bind_submits(self, c: float, js) -> list:
        """The (job, vm, arrival) submits of jobs bound to this DC at the step starting at clock
        c: VMs chosen in (due, arrival, id) order, as the visible window orders them, and the
        submits in the broker's (arrival, id) order."""
        jobs = self.jobs
        bound_pes, vm = {}, {}
        for j in sorted(js, key=lambda j: (jobs[j].due, jobs[j].a, jobs[j].id)):
            v = self.select_vm(jobs[j].cores, bound_pes)
            if v is None:
                raise ValueError(f"job {jobs[j].id} fits no VM of {self.spec.name}")
            bound_pes[v] = bound_pes.get(v, 0) + jobs[j].cores
            vm[j] = v
        return [(j, vm[j], c + (max(jobs[j].a - c, 0.0) + self.spec.net_delay))
                for j in sorted(js, key=lambda j: (jobs[j].a, jobs[j].id))]

    def submit(self, j: int, v: int, t_arrive: float) -> None:
        """A job leaves the broker for VM v (CloudSimProxy.tryToSubmitJobs + the broker's send)."""
        self.inflight[j] = (v, t_arrive)
        self.vm_of[j] = v
        heapq.heappush(self.heap, (t_arrive, TAG_SUBMIT, self.serial, j, v))
        self.serial += 1

    # ── P2 ──
    def run_step(self, c: float, T: float, submits=()) -> None:
        """Run the step from clock c to target T. submits: this step's (job, vm, arrival), in the
        broker's (arrival, id) order."""
        heap = self.heap
        self.times = set()
        self.new_finished = []
        while heap and heap[0][0] == c:       # the batch left over from the previous step
            self._process(heapq.heappop(heap))
        for j, v, t_arrive in submits:
            self.submit(j, v, t_arrive)
        while heap and heap[0][0] < T:
            self._process(heapq.heappop(heap))
        self.step += 1

    def _process(self, ev) -> None:
        t, tag, _, j, v = ev
        self.times.add(t)
        if tag == TAG_SUBMIT:
            self._on_submit(t, j, v)
        else:
            self._on_update(t)

    # ── P3 ──
    def _on_submit(self, t: float, j: int, v: int) -> None:
        del self.inflight[j]                  # the cloudlet leaves INSTANTIATED
        job = self.jobs[j]
        cle = Cle(j, job.cores, job.L)
        if self.spec.vm_pes[v] - self.alloc[v] >= job.cores:
            self._to_exec(v, cle, t)
            est = abs(job.L / self.spec.vm_mips[v])          # no file transfer time
            if est > 0.0 and not math.isinf(est):
                heapq.heappush(self.heap, (t + est, TAG_UPDATE, self.serial, -1, -1))
                self.serial += 1
        else:
            self.wait[v].append(cle)

    def _to_exec(self, v: int, cle: Cle, t: float) -> None:
        """CloudletSchedulerAbstract.addCloudletToExecList (the INEXEC status sets the start)."""
        cle.last = t
        self.exec[v].append(cle)
        self.alloc[v] += cle.pes
        if cle.j not in self.start:
            self.start[cle.j] = t

    # ── P4 ──
    def _on_update(self, t: float) -> None:
        cfg = self.cfg
        if not (t < FIRST_UPDATE_WINDOW or t >= self.last_process + cfg.min_t):
            return
        nxt = DOUBLE_MAX
        for v in range(self.spec.n_vms):      # one VM per host: HostAbstract.updateProcessing
            d = self._vm_update(v, t)
            if d > 0:
                nxt = min(d, nxt)
        if nxt != 0:
            nxt = max(nxt, cfg.update_floor)
        if nxt != DOUBLE_MAX:
            heapq.heappush(self.heap, (t + nxt, TAG_UPDATE, self.serial, -1, -1))
            self.serial += 1
        self.last_process = t

    # ── P5 ──
    def _vm_update(self, v: int, t: float) -> float:
        ex, wt = self.exec[v], self.wait[v]
        if not ex and not wt:
            return DOUBLE_MAX
        min_t = self.cfg.min_t
        mips = self.spec.vm_mips[v]
        used_mips = float(int(mips))          # (long) getAllocatedMipsForCloudlet
        finish = self.finish
        size_before = len(wt)
        nxt = DOUBLE_MAX
        finished = False
        for cle in ex:                        # updateCloudletsProcessing
            partial = int(used_mips * (t - cle.last))
            if partial != 0:                  # CloudletExecution.updateProcessing
                L = cle.L
                cle.done += partial
                fin = cle.fin
                cle.fin = fin + min(partial, L - (fin if fin < L else L))
                if cle.fin >= L and cle.j not in finish:
                    finish[cle.j] = t
                    self.new_finished.append(cle.j)
                    finished = True
            rem = cle.L - cle.done
            est = (rem if rem > 0 else 0) / mips             # cloudletEstimatedFinishTime
            if est < min_t:
                est = min_t
            cle.last = t                      # setLastOverSubscriptionDelay(0)
            if est < nxt:
                nxt = est
        if wt:                                # moveNextCloudletsFromWaitingToExecList
            pes = self.spec.vm_pes[v]
            nf = DOUBLE_MAX
            i = 0
            while i < len(wt):
                if pes - self.alloc[v] >= wt[i].pes:
                    cle = wt.pop(i)
                    self._to_exec(v, cle, t)
                    rem = cle.L - cle.done
                    est = (rem if rem > 0 else 0) / mips
                    if est < min_t:
                        est = min_t
                    if est < nf:
                        nf = est
                else:
                    i += 1
            if nf < nxt:
                nxt = nf
        if finished:                          # addCloudletsToFinishedList
            keep = []
            for x in ex:
                if x.j in finish:
                    self.alloc[v] -= x.pes
                else:
                    keep.append(x)
            ex[:] = keep
        if len(wt) != size_before and nxt == DOUBLE_MAX:   # OptimizedCloudletScheduler
            nxt = min_t
        if nxt == DOUBLE_MAX:
            return DOUBLE_MAX
        decimals = t - int(t)                 # VmAbstract.cloudletsProcessing
        return nxt if nxt - decimals < 0 else nxt - decimals

    # ── P9 ──
    def host_rows(self, now: float) -> list[list[int]]:
        """[dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts] per host at time now."""
        spec, jobs, interval = self.spec, self.jobs, self.cfg.interval
        rows = []
        for v in range(spec.n_vms):
            mips, cap = spec.vm_mips[v], spec.vm_pes[v]
            used = sum(x.pes for x in self.exec[v]) + sum(x.pes for x in self.wait[v])
            backlog = 0.0
            backlog += java_dsum(x.pes * max(0.0, x.L - (now - self.start[x.j]) * mips) / mips
                                 for x in self.exec[v])
            backlog += java_dsum((x.pes * x.L) / mips for x in self.wait[v])
            for j, (vv, _) in self.inflight.items():
                if vv == v:
                    used += jobs[j].cores
                    backlog += (jobs[j].cores * jobs[j].L) / mips
            rows.append([spec.index + 1, spec.type_id, cap, max(0, cap - used),
                         int(math.ceil(backlog / interval))])
        return rows


def simulate_dc(dc: DcState, binds, from_snapshot: DcState | None = None, until: float | None = None,
                reference: dict | None = None, record: dict | None = None,
                snapshots: dict | None = None) -> dict:
    """Replay the jobs bound to one DC and return {job index: (start, finish)}.

    dc: the DC to simulate, used as the starting state unless from_snapshot is given; neither is
        modified (a copy is simulated).
    binds: [(job index, step)], each job bound at the start of that step (>= the state's step).
        Same-step binds choose their VMs in (due, arrival, id) order, as the visible window
        orders them, and are submitted in (arrival, id) order.
    from_snapshot: the DC's state at the start of a step, before that step's binds.
    until: stop once a step would start at or after this clock (default: once every job bound
        to the DC has finished and nothing is pending).
    reference: the `record` of an earlier run of the same DC with the same `until`; once every
        bind has been applied and the state at a step boundary equals that run's, the rest of the
        run is taken over from it (the result is the same as simulating on).
    record: a dict to fill with {"keys": {step: DcState.key()}, "finish": results, "until": until};
        a run that took over from a reference also takes over its later keys, so a record can
        serve as the reference of the next run.
    snapshots: a dict to fill with {step: DcState copy} at every step start before the binds, up
        to the end of the run or the step it took over from a reference.
    Every job submitted to the DC maps to (start or None, finish or None).
    """
    if reference is not None and reference.get("until") != until:
        raise ValueError(f"the reference ran until {reference.get('until')}, this run until {until}")
    state = (from_snapshot if from_snapshot is not None else dc).copy()
    cfg, jobs = state.cfg, state.jobs
    by_step = {}
    for j, s in binds:
        if s < state.step:
            raise ValueError(f"job {jobs[j].id} bound at step {s}, before the state's step {state.step}")
        if s == 0:
            raise ValueError(f"job {jobs[j].id} bound at step 0, where the jar places nothing "
                             f"(see Episode.bind)")
        by_step.setdefault(s, []).append(j)
    last_bind = max(by_step) if by_step else -1
    c = cfg.step_clock(state.step)
    while True:
        s = state.step
        if until is not None and c >= until:
            break
        if s > last_bind and not state.busy():
            break
        if record is not None:
            record.setdefault("keys", {})[s] = state.key()
        if reference is not None and s > last_bind and reference["keys"].get(s) == state.key():
            out = {j: (state.start.get(j), state.finish.get(j)) for j in state.vm_of}
            for j, (_, fin) in out.items():
                if fin is None:
                    out[j] = reference["finish"][j]
            if record is not None:
                record["keys"].update((k, key) for k, key in reference["keys"].items() if k > s)
                record.update(finish=out, until=until)
            return out
        if snapshots is not None:
            snapshots[s] = state.copy()
        T = cfg.interval if s == 0 else c + cfg.interval
        state.run_step(c, T, state.bind_submits(c, by_step[s]) if s in by_step else ())
        c = T
    out = {j: (state.start.get(j), state.finish.get(j)) for j in state.vm_of}
    if record is not None:
        record.update(finish=out, until=until)
    return out


# ─── The whole episode ───────────────────────────────────────────────────────

class Episode:
    """One episode of the gateway, step by step: visible(), bind(), advance(), ledger_step().

    params: the environment's params (datacenters with connect_to as indices); jobs: the episode's
    jobs_json (string or list), in the order the simulator receives it. horizon overrides
    params["max_episode_length"].
    """

    def __init__(self, params: dict, jobs, horizon: int | None = None):
        cfg = self.cfg = Settings(params, horizon)
        self.specs = [DcSpec(i, dc, cfg) for i, dc in enumerate(params["datacenters"])]
        self.jobs = parse_jobs(jobs, cfg)
        self.by_id = {job.id: job.idx for job in self.jobs}
        if len(self.by_id) != len(self.jobs):
            raise ValueError("duplicate job ids")
        if any(job.a >= cfg.horizon * cfg.interval for job in self.jobs):   # WrappedSimulation.reset
            raise ValueError("a job arrives at or after the horizon")
        self.dcs = [DcState(spec, self.jobs, cfg) for spec in self.specs]
        self.clock = cfg.min_t                # after the proxy's proceedClockTo(minTimeBetweenEvents)
        self.first = True
        self.step_index = 0                   # steps taken (WrappedSimulation.currentStep)
        self.arrival_order = sorted(range(len(self.jobs)), key=lambda i: (self.jobs[i].a, self.jobs[i].id))
        self._arr_ptr = 0
        self.pool = set()                     # arrived, neither submitted nor evicted (jobQueue)
        self.bound = {}                       # job -> (dc, vm)
        self.bind_step = {}                   # job -> step it was bound at
        self.cost = {}                        # job -> c
        self.evicted = set()
        # SlaLedger. A job resolves once: at the step it finishes, or at the first step that ends
        # past its due unless it is running on time; each step settles its jobs in job-list
        # order, as Java iterates.
        self.resolved = [False] * len(self.jobs)
        self.n_unresolved = len(self.jobs)
        self._overdue_order = sorted(range(len(self.jobs)), key=lambda i: (self.jobs[i].due, i))
        self._overdue_ptr = 0
        self._new_finished = []
        self.offered_value = 0.0
        for job in self.jobs:
            self.offered_value += job.V
        self.outcome = {}                     # job -> (met, step it resolved at)
        self.terminated = False
        self._placed = 0
        self.last_info = None

    def copy(self) -> "Episode":
        """An independent branch of the episode (DC states copied, static tables shared)."""
        c = Episode.__new__(Episode)
        c.__dict__.update(self.__dict__)
        c.dcs = [dc.copy() for dc in self.dcs]
        for name in ("pool", "bound", "bind_step", "cost", "evicted", "outcome"):
            setattr(c, name, type(getattr(self, name))(getattr(self, name)))
        c.resolved = list(self.resolved)
        c._new_finished = list(self._new_finished)
        c.last_info = dict(self.last_info) if self.last_info else None
        return c

    # ── P7 ──
    def target(self) -> float:
        return self.cfg.interval if self.first else self.clock + self.cfg.interval

    def _admit(self, T: float) -> None:
        jobs, order = self.jobs, self.arrival_order
        while self._arr_ptr < len(order) and jobs[order[self._arr_ptr]].a < T:
            i = order[self._arr_ptr]
            if i not in self.evicted:
                self.pool.add(i)
            self._arr_ptr += 1

    def visible_idx(self) -> list[int]:
        """Job indices in slot order at the current step."""
        self._admit(self.target())
        jobs = self.jobs
        return sorted(self.pool, key=lambda i: (jobs[i].due, jobs[i].a, jobs[i].id))[: self.cfg.max_jobs_waiting]

    def visible(self, s: int | None = None) -> list[int]:
        """Job ids in slot order at step s (default: the current step)."""
        if s is not None and s != self.step_index:
            raise ValueError(f"the episode is at step {self.step_index}, not {s}")
        return [self.jobs[i].id for i in self.visible_idx()]

    # ── binds (WrappedSimulation.executeRlCloudletToDcAction) ──
    def bind(self, slot_actions) -> int:
        """One action per visible slot (k > 0 places the slot's job on DC k - 1); call once per
        step, before advance(). Returns the number of jobs placed."""
        if self.terminated:
            raise RuntimeError("the episode is over")
        vis = self.visible_idx()
        if len(slot_actions) < len(vis):
            raise ValueError(f"{len(slot_actions)} actions for {len(vis)} visible jobs")
        if self.first:
            # The broker lists its VMs only once it processes their creation acks, which are due
            # at min_time_between_events, the clock the first step starts at, so they run during
            # that step's clock advance: every bind of the first step finds no VM, and Java warns
            # and skips it (a job arriving before the first target stays unplaced until step 1).
            return 0
        bound_pes = [dict() for _ in self.dcs]
        placed = 0
        for i, j in enumerate(vis):
            k = int(slot_actions[i])
            if k == 0 or j in self.bound:
                continue
            d = k - 1
            if not 0 <= d < len(self.dcs):
                continue                      # no VM has that DC id: Java warns and skips it
            v = self.dcs[d].select_vm(self.jobs[j].cores, bound_pes[d])
            if v is None:
                continue                      # no VM can hold it: Java warns and skips it
            self.bound[j] = (d, v)
            self.bind_step[j] = self.step_index
            self.cost[j] = self.jobs[j].cost_on(self.specs[d], self.cfg)
            bound_pes[d][v] = bound_pes[d].get(v, 0) + self.jobs[j].cores
            placed += 1
        self._placed += placed
        return placed

    # ── P2, P6 ──
    def advance(self) -> dict:
        """Run the step (and at the horizon the drain); returns ledger_step()."""
        if self.terminated:
            raise RuntimeError("the episode is over")
        self.step_index += 1
        T = self.target()
        self._admit(T)
        jobs_waiting = len(self.pool)
        self._run_one_timestep(T)
        sums = dict.fromkeys(("sla_value_realized", "sla_penalty_paid", "resource_cost"), 0.0)
        sums.update(jobs_met=0, jobs_violated=0, jobs_expired_unplaced=0)
        self._resolve(sums)
        if self.step_index >= self.cfg.horizon:
            self._drain(sums)
        self.terminated = self.n_unresolved == 0
        Z = self.offered_value
        sums.update(
            jobs_waiting=jobs_waiting, jobs_placed=self._placed, offered_value=Z,
            unshaped_reward=((sums["sla_value_realized"] - sums["sla_penalty_paid"]
                              - sums["resource_cost"]) / Z if Z > 0 else 0.0),
            terminated=self.terminated, step=self.step_index,
        )
        self._placed = 0
        self.last_info = sums
        return sums

    def step(self, slot_actions) -> dict:
        self.bind(slot_actions)
        return self.advance()

    def ledger_step(self) -> dict | None:
        """The last step's ledger: SLA value, penalty, cost, met, violated, expired unplaced,
        placed, waiting, the unshaped reward, terminated."""
        return self.last_info

    def _run_one_timestep(self, T: float) -> None:
        c, jobs = self.clock, self.jobs
        submits = [[] for _ in self.dcs]
        for j in sorted((i for i in self.pool if i in self.bound), key=lambda i: (jobs[i].a, jobs[i].id)):
            d, v = self.bound[j]
            submits[d].append((j, v, c + (max(jobs[j].a - c, 0.0) + self.specs[d].net_delay)))
            self.pool.discard(j)
        times = set()
        for dc, sub in zip(self.dcs, submits):
            dc.run_step(c, T, sub)
            times |= dc.times
            self._new_finished += dc.new_finished
            if dc.heap and dc.heap[0][0] == T:
                times.add(T)
        # proceedClockTo gives up after MAX_CLOCK_ITERATIONS runFor calls, one per event batch:
        # a step that needs more would end early in Java.
        if 2 * len(times) + 4 >= MAX_CLOCK_ITERATIONS:
            raise PortError(f"step {self.step_index}: about {2 * len(times)} event batches, near "
                            f"proceedClockTo's cutoff of {MAX_CLOCK_ITERATIONS}")
        self.clock = T
        self.first = False

    def _settle(self, sums: dict, j: int, met: bool) -> None:
        job = self.jobs[j]
        if met:
            sums["sla_value_realized"] += job.V
            sums["jobs_met"] += 1
        else:
            sums["sla_penalty_paid"] += job.P
            sums["jobs_violated"] += 1
        sums["resource_cost"] += self.cost.get(j, 0.0)
        self.outcome[j] = (met, self.step_index)

    def finish_time(self, j: int) -> float | None:
        b = self.bound.get(j)
        return None if b is None else self.dcs[b[0]].finish.get(j)

    def on_time(self, j: int) -> bool:
        """SlaLedger.onTime: the job's execution ends by its due (executionEnd = start + L / vm
        MIPS, float for float as Java computes it); never for a job that has not started."""
        b = self.bound.get(j)
        start = None if b is None else self.dcs[b[0]].start.get(j)
        if start is None:
            return False
        job = self.jobs[j]
        return start + job.L / self.specs[b[0]].vm_mips[b[1]] <= job.due + ON_TIME_EPSILON

    def _resolve(self, sums: dict) -> None:
        """SlaLedger.resolve(clock) and the eviction of the unplaced jobs it expires. A running
        job that is on time is passed over here and resolved at the step that records its
        finish (it is then among that step's finished jobs)."""
        now, jobs, order = self.clock, self.jobs, self._overdue_order
        candidates = set(self._new_finished)
        self._new_finished = []
        while self._overdue_ptr < len(order) and jobs[order[self._overdue_ptr]].due < now:
            candidates.add(order[self._overdue_ptr])
            self._overdue_ptr += 1
        for j in sorted(candidates):
            if self.resolved[j]:
                continue
            if self.finish_time(j) is not None:
                self._settle(sums, j, self.on_time(j))
            elif now > jobs[j].due and not self.on_time(j):
                self._settle(sums, j, False)
                if j not in self.bound:
                    sums["jobs_expired_unplaced"] += 1
                    self.pool.discard(j)
                    self.evicted.add(j)
            else:
                continue
            self.resolved[j] = True
            self.n_unresolved -= 1

    def _drain(self, sums: dict) -> None:
        for j in range(len(self.jobs)):                   # SlaLedger.expireAllUnplaced
            if not self.resolved[j] and j not in self.bound:
                self._settle(sums, j, False)
                sums["jobs_expired_unplaced"] += 1
                self.pool.discard(j)
                self.evicted.add(j)
                self.resolved[j] = True
                self.n_unresolved -= 1
        while self.n_unresolved:
            self._run_one_timestep(self.clock + self.cfg.interval)
            self._resolve(sums)

    # ── P9 ──
    def infra_obs(self) -> list[int]:
        """The infrastructure observation's real rows, flattened (JobPlacementEnv pads it). The
        jar's last observation of an episode in which every job finished may list none (P9)."""
        out = []
        for dc in self.dcs:
            for row in dc.host_rows(self.clock):
                out.extend(row)
        return out

    def jobs_obs(self) -> list[int]:
        """The wire format: [cores, location, nominal_runtime_ref, time_to_due, s0, s1, s2] per slot."""
        interval, now = self.cfg.interval, self.clock
        out = []
        for j in self.visible_idx():
            job = self.jobs[j]
            one_hot = [0, 0, 0]
            one_hot[job.sens] = 1
            out += [job.cores, job.loc, job.r, int(max(0.0, math.floor((job.due - now) / interval)))] + one_hot
        return out

    def results(self) -> dict:
        """{job id: (dc index or None, vm or None, start, finish, met or None)}."""
        out = {}
        for job in self.jobs:
            d, v = self.bound.get(job.idx, (None, None))
            dc = self.dcs[d] if d is not None else None
            out[job.id] = (d, v, dc.start.get(job.idx) if dc else None,
                           dc.finish.get(job.idx) if dc else None,
                           self.outcome[job.idx][0] if job.idx in self.outcome else None)
        return out

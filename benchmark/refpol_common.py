"""Shared pieces of the RING-N reference policies: static DC facts and one worker's observation.

Action k != 0 places a job on datacenters[k - 1], whose hosts carry dc_id k in the observation;
action 0 is the no-op. Every decision key the policies use is built from physical DC attributes
and ends in the DC name, so a relabelled topology (PI-S renumbers S's DCs but keeps their
names) gets the same choices. A relabelling that renamed DCs could swap two exactly symmetric
DCs; PI-S does not rename.
"""
from dataclasses import dataclass

import numpy as np

TYPE_ID = {"cloud": 1, "edge": 2, "micro": 3}              # the observation's dc_type; 0 pads
SENSITIVITIES = ("tolerant", "moderate", "critical")        # simulator levels 0, 1, 2 = s0, s1, s2
HOST_FEATURES = 5   # dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts
JOB_FEATURES = 6    # cores, nominal_runtime_ref, time_to_due, s0, s1, s2
OBS_KEYS = ("infrastructure_state", "jobs_waiting_state", "reach_mask")


@dataclass(frozen=True)
class Dc:
    k: int                  # action index = obs dc_id; the DC is datacenters[k - 1]
    name: str
    type: str
    type_id: int
    nd: float               # network delay, timesteps
    kappa: float            # cost per reference core-second
    vm_pes: tuple           # per host, in params host order (one VM per host)
    vm_mips: tuple
    capacity: int           # sum of VM PEs
    max_vm_pes: int
    connect_to: tuple       # action indices


class Topology:
    """Static per-action DC facts from env params, plus the reward and shape constants."""

    def __init__(self, dcs: list, V: tuple, P: tuple, mips_ref: float, interval: float,
                 H: int, n_slots: int, n_actions: int):
        self.dcs = dcs
        self.ks = [dc.k for dc in dcs]
        self.V, self.P = V, P
        self.mips_ref, self.interval, self.H = mips_ref, interval, H
        self.n_slots, self.n_actions = n_slots, n_actions
        self.params = None                          # the env params, when built from them
        self._by_k = {dc.k: dc for dc in dcs}
        self._static = {dc.k: (dc.type_id, -dc.capacity, dc.name) for dc in dcs}

    @classmethod
    def from_params(cls, params: dict) -> "Topology":
        dcs = []
        for idx, dc in enumerate(params["datacenters"]):
            vm_pes, vm_mips = [], []
            for host in dc["hosts"]:
                for _ in range(int(host.get("amount", 1))):
                    for vm in host["vms"]:
                        for _ in range(int(vm.get("amount", 1))):
                            vm_pes.append(int(vm["pes"]))
                            vm_mips.append(float(vm["pe_mips"]))
            dc_type = dc["type"]
            dcs.append(Dc(k=idx + 1, name=dc["name"], type=dc_type, type_id=TYPE_ID[dc_type],
                          nd=float(params[f"network_delay_{dc_type}"]),
                          kappa=float(params[f"cost_{dc_type}"]),
                          vm_pes=tuple(vm_pes), vm_mips=tuple(vm_mips), capacity=sum(vm_pes),
                          max_vm_pes=max(vm_pes),
                          connect_to=tuple(int(c) + 1 for c in dc.get("connect_to", []))))
        topo = cls(dcs,
                   V=tuple(float(params[f"sla_value_{s}"]) for s in SENSITIVITIES),
                   P=tuple(float(params[f"sla_penalty_{s}"]) for s in SENSITIVITIES),
                   mips_ref=float(params["mips_ref"]), interval=float(params["timestep_interval"]),
                   H=int(params["max_episode_length"]), n_slots=int(params["max_jobs_waiting"]),
                   n_actions=int(params["max_datacenters"]))
        topo.params = params
        return topo

    def dc(self, k: int) -> Dc:
        return self._by_k[k]

    def static_key(self, k: int) -> tuple:
        """Tier, capacity descending, name."""
        return self._static[k]

    def dyn_key(self, k: int, free: float) -> tuple:
        """Free PEs descending, then the static key."""
        return (-free,) + self._static[k]

    def legal_set(self, origin: int) -> frozenset:
        """Actions a job from DC `origin` may take under the topology (JobPlacementEnv's reach):
        its own DC and the DCs it connects to, or every DC if it connects to none."""
        dc = self._by_k[origin]
        if not dc.connect_to:
            return frozenset(self.ks)
        return frozenset((origin,) + dc.connect_to)


class ObsView:
    """One worker's observation and action mask.

    Hosts: rows with dc_id > 0 grouped by dc_id in row order, as Python lists per DC (cap,
    free, backlog). Jobs: per slot cores, r (nominal runtime at mips_ref, timesteps), ttd
    (timesteps to due), sensitivity level and its V and P; `real` is cores > 0.
    reach and mask are [n_slots, n_actions] bool.
    """

    def __init__(self, obs_i: dict, mask_i, topo: Topology):
        self.topo = topo
        hosts = np.asarray(obs_i["infrastructure_state"]).reshape(-1, HOST_FEATURES)
        dc_ids = hosts[:, 0]
        if int((dc_ids > 0).sum()) != sum(len(dc.vm_pes) for dc in topo.dcs):
            raise ValueError(f"observation has {int((dc_ids > 0).sum())} host rows, the topology "
                             f"{sum(len(dc.vm_pes) for dc in topo.dcs)}")
        self.cap, self.free, self.backlog = {}, {}, {}
        for dc in topo.dcs:
            rows = hosts[dc_ids == dc.k]
            cap = tuple(rows[:, 2].tolist())
            if cap != dc.vm_pes:
                raise ValueError(f"DC {dc.name} (dc_id {dc.k}): observed vm_capacity_pes {cap}, "
                                 f"params say {dc.vm_pes}")
            self.cap[dc.k] = cap
            self.free[dc.k] = rows[:, 3].tolist()
            self.backlog[dc.k] = rows[:, 4].tolist()

        jobs = np.asarray(obs_i["jobs_waiting_state"]).reshape(topo.n_slots, JOB_FEATURES)
        onehot = jobs[:, 3:6]
        self.real = jobs[:, 0] > 0
        if (onehot[self.real].sum(axis=1) != 1).any() or onehot[~self.real].any():
            raise ValueError("every real job slot needs exactly one sensitivity bit, padding none")
        self.cores = jobs[:, 0].tolist()
        self.r = jobs[:, 1].tolist()
        self.ttd = jobs[:, 2].tolist()
        self.sens = onehot.argmax(axis=1).tolist()
        self.V = [topo.V[s] for s in self.sens]
        self.P = [topo.P[s] for s in self.sens]
        self.slots = np.flatnonzero(self.real).tolist()
        self.reach = np.asarray(obs_i["reach_mask"]).reshape(topo.n_slots, topo.n_actions).astype(bool)
        self.mask = np.asarray(mask_i).reshape(topo.n_slots, topo.n_actions).astype(bool)

    def legal(self, i: int) -> list:
        """The DC actions the mask allows for slot i (the no-op excluded)."""
        return [k for k in self.topo.ks if self.mask[i, k]]

    def reach_set(self, i: int) -> frozenset:
        return frozenset(k for k in self.topo.ks if self.reach[i, k])

    def run(self, i: int, k: int, h: int) -> float:
        """Slot i's runtime on host h of DC k, in timesteps."""
        return self.r[i] * self.topo.mips_ref / self.topo.dc(k).vm_mips[h]

    def cost(self, i: int, k: int) -> float:
        """kappa * cores * r = kappa * cores * mi / mips_ref, the ledger's placement cost."""
        return self.topo.dc(k).kappa * self.cores[i] * self.r[i]

    def density(self, i: int) -> float:
        return (self.V[i] + self.P[i]) / (self.cores[i] * max(self.r[i], 1e-9))

"""Generate and audit the RING-N topology family (docs/analysis/05-benchmark-redesign-ring-n.json).

    python3 benchmark/gen_ring_family.py          # write common/topologies/ring/*.yml + manifest.json
    python3 benchmark/gen_ring_family.py --audit  # check the written family and print its table

One cloud DC plus a ring of N-1 origins alternating edge, micro, edge, micro, ... Each origin connects
to its two ring neighbours and the cloud, so a job from ring position p may go to {p, p-1, p+1, cloud}.

Deviation from the spec: the spec keeps a 4-host cloud in every member, so the cloud's PE share falls
from 31% (N=7) to 13% (N=19) and cloud scarcity becomes a confound shared by both count arms. Here the
cloud host count is derived from each member's non-cloud capacity so its share tracks S's (21.05%).
"""

import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RING_DIR = os.path.join(REPO, "common", "topologies", "ring")
RL_MANAGER = os.path.join(REPO, "common", "rl-manager")
sys.path[:0] = [RL_MANAGER, os.path.join(RL_MANAGER, "gym_cloudsimplus")]  # audit uses the repo loader
MANIFEST_NAME = "manifest.json"
MANIFEST_KEYS = ["id", "yaml", "n_dcs", "origins", "capacity_pes", "tier_pe_share", "load",
                 "relabel_of", "permutation_seed", "isolates"]  # shared contract with the workload builder

# Shape constants (spec: shape_constants). Action 0 is the no-op, so 23 real DCs are addressable.
MAX_DATACENTERS = 24
MAX_HOSTS = 8
TOTAL_HOSTS = 192
MAX_CLOUD_SHARE_SPREAD = 0.03
RING_DEGREE = 4  # own DC, two ring neighbours, cloud

# Identical in every member; RAM/storage/bw reuse euromlsys_c.yml and never bind. One VM per host.
TIERS = {
    "cloud": {"pes": 32, "pe_mips": 100, "ram": 65536, "storage": 4000000, "bw": 100000},
    "edge": {"pes": 16, "pe_mips": 60, "ram": 65536, "storage": 2000000, "bw": 10000},
    "micro": {"pes": 8, "pe_mips": 40, "ram": 65536, "storage": 1000000, "bw": 1000},
}
RING_TYPES = ("edge", "micro")  # ring position 1 is an edge, then they alternate
CLOUD_NAME = "cloud_dc"

ANCHOR_N = 11
ANCHOR_HOSTS = 4  # S has 4 hosts in every DC, cloud included

# Cloud hosts are derived (cloud_hosts); everything else is a design choice:
#   C1: non-cloud DCs keep S's 4 hosts, load rho=1 so lambda scales with capacity.
#   C2: non-cloud hosts chosen so capacity is identical across the arm (632 PEs, 1.04x S) and runs
#       S's job instance. N=19 hits the integer floor: e2/m3 is the only in-window option near 632,
#       so its edge:micro mix shifts ~7 pp toward micro.
#   LOCK: c3/e2/m3 (488 PEs, 0.80x S). The spec's N=15 h=3 LOCK is isomorphic to C2-N15 once the
#       cloud is corrected; the lockbox must not share a topology with a development member.
# id, N, edge hosts/DC, micro hosts/DC, load, relabel_of, permutation_seed, isolates
MEMBERS = [
    ("S", 11, 4, 4, {"rho": 1.0}, None, None, "anchor"),
    ("C1-N7", 7, 4, 4, {"rho": 1.0}, None, None, "count @ fixed per-DC size"),
    ("C1-N15", 15, 4, 4, {"rho": 1.0}, None, None, "count @ fixed per-DC size"),
    ("C1-N19", 19, 4, 4, {"rho": 1.0}, None, None, "count @ fixed per-DC size"),
    ("C2-N7", 7, 7, 7, {"lambda_of": "S"}, None, None, "count @ fixed total capacity"),
    ("C2-N15", 15, 3, 3, {"lambda_of": "S"}, None, None, "count @ fixed total capacity"),
    ("C2-N19", 19, 2, 3, {"lambda_of": "S"}, None, None,
     "count @ fixed total capacity; edge:micro mix shifted (integer floor)"),
    ("GAM-lo", 11, 3, 3, {"lambda_of": "S"}, None, None, "capacity regime (less capacity)"),
    ("GAM-hi", 11, 6, 6, {"lambda_of": "S"}, None, None, "capacity regime (more capacity)"),
    ("PI-S", 11, 4, 4, {"lambda_of": "S"}, "S", 1, "relabelling control (isomorphic to S)"),
    ("PI-N19", 19, 4, 4, {"lambda_of": "C1-N19"}, "C1-N19", 2,
     "relabelling control (isomorphic to C1-N19)"),
    ("LOCK", 15, 2, 3, {"rho": 1.22}, None, 3, "lockbox: compound count+capacity+tier-mix+load+relabel"),
]


# ─── Construction ───────────────────────────────────────────────────────────

def _non_cloud_pes(n_dcs: int, edge_hosts: int, micro_hosts: int) -> int:
    per_type = (n_dcs - 1) // len(RING_TYPES)
    return per_type * (edge_hosts * TIERS["edge"]["pes"] + micro_hosts * TIERS["micro"]["pes"])


def cloud_hosts(n_dcs: int, edge_hosts: int, micro_hosts: int) -> int:
    """Largest cloud whose PE share does not exceed S's.

    Rounding to nearest instead would put C1-N15 at 22.2% against C1-N7's forced 18.2%, a 4.0 pp
    spread; rounding down keeps every member in [18.2%, 21.1%]. Integer arithmetic, no float floor.
    """
    anchor_cloud = ANCHOR_HOSTS * TIERS["cloud"]["pes"]
    anchor_rest = _non_cloud_pes(ANCHOR_N, ANCHOR_HOSTS, ANCHOR_HOSTS)
    rest = _non_cloud_pes(n_dcs, edge_hosts, micro_hosts)
    return anchor_cloud * rest // (anchor_rest * TIERS["cloud"]["pes"])


def ring_datacenters(n_dcs: int, edge_hosts: int, micro_hosts: int) -> list[dict]:
    """Cloud first, then ring positions 1..N-1 in order."""
    assert (n_dcs - 1) % len(RING_TYPES) == 0, f"ring of {n_dcs - 1} cannot alternate edge/micro"
    hosts = {"cloud": cloud_hosts(n_dcs, edge_hosts, micro_hosts),
             "edge": edge_hosts, "micro": micro_hosts}
    types = [RING_TYPES[i % len(RING_TYPES)] for i in range(n_dcs - 1)]
    names = [f"{t}_dc_{p:02d}" for p, t in enumerate(types, start=1)]
    dcs = [{"name": CLOUD_NAME, "type": "cloud", "hosts": hosts["cloud"], "connect_to": []}]
    for i, (name, dc_type) in enumerate(zip(names, types)):
        neighbours = [names[i - 1], names[(i + 1) % len(names)]]
        dcs.append({"name": name, "type": dc_type, "hosts": hosts[dc_type],
                    "connect_to": neighbours + [CLOUD_NAME]})
    return dcs


def _render_yaml(member_id: str, dcs: list[dict]) -> str:
    lines = [f"# RING-N member {member_id}. Generated by benchmark/gen_ring_family.py; do not edit."]
    for dc in dcs:
        tier = TIERS[dc["type"]]
        lines += ["- !datacenter", f"  name: {dc['name']}", f"  type: {dc['type']}", "  amount: 1"]
        if dc["connect_to"]:
            lines += ["  connect_to:"] + [f"    - {name}" for name in dc["connect_to"]]
        lines += [
            "  hosts:",
            "    - !host",
            f"      amount: {dc['hosts']}",
            f"      pes: {tier['pes']}",
            f"      pe_mips: {tier['pe_mips']}",
            f"      ram: {tier['ram']}",
            f"      storage: {tier['storage']}",
            f"      bw: {tier['bw']}",
            "      vms:",
            "        - !vm",
            "          amount: 1",
            f"          pes: {tier['pes']}",
            f"          pe_mips: {tier['pe_mips']}",
            f"          ram: {tier['ram']}",
            f"          size: {tier['storage']}",
            f"          bw: {tier['bw']}",
        ]
    return "\n".join(lines) + "\n"


def _capacity_by_tier(dcs: list[dict]) -> dict:
    """PEs per tier of DCs as the repo loader returns them."""
    pes = {t: 0 for t in TIERS}
    for dc in dcs:
        pes[dc["type"]] += sum(h["amount"] * h["pes"] for h in dc["hosts"])
    return pes


def write_family(ring_dir: str = RING_DIR) -> list[dict]:
    os.makedirs(ring_dir, exist_ok=True)
    manifest = []
    for member_id, n_dcs, edge, micro, load, relabel_of, seed, isolates in MEMBERS:
        dcs = ring_datacenters(n_dcs, edge, micro)
        origins = [dc["name"] for dc in dcs[1:]]  # ring order: identical for a relabelling
        if seed is not None:
            order = np.random.default_rng(seed).permutation(len(dcs))
            dcs = [dcs[i] for i in order]
        with open(os.path.join(ring_dir, f"{member_id}.yml"), "w") as f:
            f.write(_render_yaml(member_id, dcs))
        pes = {t: 0 for t in TIERS}
        for dc in dcs:
            pes[dc["type"]] += dc["hosts"] * TIERS[dc["type"]]["pes"]
        capacity = sum(pes.values())
        manifest.append({
            "id": member_id,
            "yaml": f"ring/{member_id}.yml",
            "n_dcs": n_dcs,
            "origins": origins,
            "capacity_pes": capacity,
            "tier_pe_share": {t: round(pes[t] / capacity, 4) for t in TIERS},
            "load": load,
            "relabel_of": relabel_of,
            "permutation_seed": seed,
            "isolates": isolates,
        })
    with open(os.path.join(ring_dir, MANIFEST_NAME), "w") as f:
        json.dump({"members": manifest}, f, indent=2)
        f.write("\n")
    return manifest


# ─── Audit ──────────────────────────────────────────────────────────────────

def load_topology(path: str) -> list[dict]:
    """Load a member YAML through the repo's own loader, as job-placement does."""
    from utils import misc  # heavy (torch/sb3); only the audit needs it

    misc._register_yaml_constructors()
    with open(path) as f:
        dcs = [dc.to_dict() for dc in yaml.load(f, Loader=yaml.Loader)]
    misc._check_datacenters_unique(dcs)
    misc._check_datacenter_amounts_are_one(dcs)
    return dcs


def _structure_signature(dcs: list[dict]) -> list:
    """Isomorphism-invariant: isomorphic topologies always get equal signatures, so unequal
    signatures prove two members differ (equal ones are conservatively treated as identical)."""
    label = {dc["name"]: (dc["type"], json.dumps(dc["hosts"], sort_keys=True)) for dc in dcs}
    return sorted((label[dc["name"]], sorted(label[c] for c in dc["connect_to"])) for dc in dcs)


def check_member(entry: dict, dcs: list[dict]) -> None:
    mid = entry["id"]
    names = [dc["name"] for dc in dcs]
    assert len(dcs) == entry["n_dcs"], f"{mid}: {len(dcs)} DCs, manifest says {entry['n_dcs']}"
    assert len(dcs) <= MAX_DATACENTERS - 1, f"{mid}: {len(dcs)} DCs > {MAX_DATACENTERS - 1} addressable"
    assert all(dc["amount"] == 1 for dc in dcs), f"{mid}: every DC needs amount 1"

    clouds = [dc for dc in dcs if dc["type"] == "cloud"]
    assert len(clouds) == 1, f"{mid}: {len(clouds)} cloud DCs, need exactly one"
    cloud = clouds[0]
    assert not cloud["connect_to"], f"{mid}: the cloud must have no connect_to"
    permuted = entry["permutation_seed"] is not None
    assert (names.index(cloud["name"]) != 0) == permuted, (
        f"{mid}: cloud must be at index 0 exactly when the member is not permuted")

    origins = entry["origins"]
    assert sorted(origins) == sorted(n for n in names if n != cloud["name"]), (
        f"{mid}: origins must be exactly the non-cloud DCs")
    by_name = {dc["name"]: dc for dc in dcs}
    for i, name in enumerate(origins):
        dc = by_name[name]
        assert dc["type"] == RING_TYPES[i % len(RING_TYPES)], f"{mid}: ring must alternate edge/micro"
        expected = [origins[i - 1], origins[(i + 1) % len(origins)], cloud["name"]]
        assert dc["connect_to"] == expected, (
            f"{mid}: {name} connect_to {dc['connect_to']} is not [its two ring neighbours, cloud] "
            f"{expected}")

    host_total = 0
    for dc in dcs:
        tier = TIERS[dc["type"]]
        assert len(dc["hosts"]) == 1, f"{mid}: {dc['name']} must have one host group"
        host = dc["hosts"][0]
        assert 1 <= host["amount"] <= MAX_HOSTS, f"{mid}: {dc['name']} has {host['amount']} hosts"
        assert {k: host[k] for k in tier} == tier, f"{mid}: {dc['name']} host differs from tier spec"
        vm = {"amount": 1, "pes": tier["pes"], "pe_mips": tier["pe_mips"], "ram": tier["ram"],
              "size": tier["storage"], "bw": tier["bw"]}
        assert host["vms"] == [vm], f"{mid}: {dc['name']} must have one VM filling the host"
        host_total += host["amount"]
    assert host_total <= TOTAL_HOSTS, f"{mid}: {host_total} hosts > {TOTAL_HOSTS} observation slots"

    pes = _capacity_by_tier(dcs)
    assert sum(pes.values()) == entry["capacity_pes"], f"{mid}: capacity differs from manifest"
    shares = {t: round(pes[t] / entry["capacity_pes"], 4) for t in TIERS}
    assert shares == entry["tier_pe_share"], f"{mid}: tier shares differ from manifest"
    check_legal_actions(mid, dcs, origins)


def check_legal_actions(mid: str, dcs: list[dict], origins: list[str]) -> None:
    """Every origin has exactly RING_DEGREE legal actions and every DC is selectable by some origin,
    computed with job-placement's own index translation and mask builder."""
    from utils import misc
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    indexed = misc._translate_connect_to_names_to_idx(copy.deepcopy(dcs))
    mask = JobPlacementEnv._build_location_mask(
        SimpleNamespace(max_datacenters=MAX_DATACENTERS), indexed)
    rows = mask[[misc._get_dc_idx_by_name(o, indexed) for o in origins]]
    degrees = rows.sum(axis=1)
    assert (degrees == RING_DEGREE).all(), f"{mid}: origin legal-action counts {degrees.tolist()}"
    selectable = rows.any(axis=0)
    real = slice(1, len(dcs) + 1)  # action = DC index + 1; action 0 is the no-op
    assert selectable[real].all(), f"{mid}: DC indices {np.flatnonzero(~selectable[real]).tolist()} unreachable"
    assert not selectable[0] and not selectable[len(dcs) + 1:].any(), f"{mid}: action outside real DCs"


def effective_rho(entry: dict, by_id: dict) -> float:
    load = entry["load"]
    if "rho" in load:
        return load["rho"]
    source = by_id[load["lambda_of"]]
    return source["load"]["rho"] * source["capacity_pes"] / entry["capacity_pes"]


def audit(ring_dir: str = RING_DIR) -> list[dict]:
    with open(os.path.join(ring_dir, MANIFEST_NAME)) as f:
        entries = json.load(f)["members"]
    by_id = {e["id"]: e for e in entries}
    assert len(by_id) == len(entries), "duplicate member ids"
    topologies = {}
    for entry in entries:
        assert list(entry) == MANIFEST_KEYS, f"{entry.get('id')}: manifest keys {list(entry)}"
        dcs = load_topology(os.path.join(os.path.dirname(ring_dir), entry["yaml"]))
        check_member(entry, dcs)
        topologies[entry["id"]] = dcs

    for entry in entries:
        load = entry["load"]
        assert list(load) in (["rho"], ["lambda_of"]), f"{entry['id']}: load must be rho or lambda_of"
        if "lambda_of" in load:
            assert "rho" in by_id[load["lambda_of"]]["load"], (
                f"{entry['id']}: lambda_of must name a member with an explicit rho")

    for entry in entries:
        source = entry["relabel_of"]
        if source is None:
            continue
        mine, theirs = topologies[entry["id"]], topologies[source]
        assert {dc["name"]: dc for dc in mine} == {dc["name"]: dc for dc in theirs}, (
            f"{entry['id']}: not an exact relabelling of {source}")
        assert [dc["name"] for dc in mine] != [dc["name"] for dc in theirs], (
            f"{entry['id']}: same DC order as {source}")
        assert entry["origins"] == by_id[source]["origins"], f"{entry['id']}: origins differ from {source}"

    signatures = {e["id"]: _structure_signature(topologies[e["id"]])
                  for e in entries if e["relabel_of"] is None}
    ids = list(signatures)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert signatures[a] != signatures[b], f"{a} and {b} are structurally identical"

    cloud_shares = [e["tier_pe_share"]["cloud"] for e in entries]
    spread = max(cloud_shares) - min(cloud_shares)
    assert spread <= MAX_CLOUD_SHARE_SPREAD, f"cloud PE share spread {spread:.4f} > {MAX_CLOUD_SHARE_SPREAD}"

    _print_table(entries, topologies, by_id, spread)
    return entries


def _print_table(entries: list[dict], topologies: dict, by_id: dict, spread: float) -> None:
    print(f"{'id':<8}{'N':>3}  {'hosts c/e/m':<12}{'C (PEs)':>8}  {'cloud':>6}{'edge':>7}{'micro':>7}"
          f"  {'rho_eff':>7}  load")
    for e in entries:
        hosts = {dc["type"]: dc["hosts"][0]["amount"] for dc in topologies[e["id"]]}
        share = e["tier_pe_share"]
        rule = (f"rho={e['load']['rho']}" if "rho" in e["load"] else f"lambda_of {e['load']['lambda_of']}")
        if e["permutation_seed"] is not None:
            rule += f"; permuted seed {e['permutation_seed']}"
        print(f"{e['id']:<8}{e['n_dcs']:>3}  {hosts['cloud']}/{hosts['edge']}/{hosts['micro']:<8}"
              f"{e['capacity_pes']:>8}  {100 * share['cloud']:>5.2f}%{100 * share['edge']:>6.1f}%"
              f"{100 * share['micro']:>6.1f}%  {effective_rho(e, by_id):>7.3f}  {rule}")
    print(f"cloud PE share spread: {100 * spread:.2f} pp (limit {100 * MAX_CLOUD_SHARE_SPREAD:.0f} pp)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", action="store_true", help="check the written family and print it")
    if parser.parse_args().audit:
        audit()
    else:
        write_family()
        print(f"wrote {len(MEMBERS)} members to {RING_DIR}")

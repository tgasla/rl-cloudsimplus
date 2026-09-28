"""RING-N level generator. Run: python3 -m pytest common/rl-manager/tests/test_levels.py"""

import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                  # rl-manager

from utils import levels  # noqa: E402

LEVELS_PY = os.path.join(os.path.dirname(HERE), "utils", "levels.py")
COMMON = os.path.abspath(os.path.join(HERE, "..", ".."))
RING_MANIFEST = os.path.join(COMMON, "topologies", "ring", "manifest.json")
TIERS = {"cloud": (32, 100), "edge": (16, 60), "micro": (8, 40)}         # host/VM PEs, pe_mips
VALIDATE_LEVELS = 1000
RING_VALIDATE_LEVELS = 100   # deterministic; the CLI runs the full VALIDATE_LEVELS


@pytest.fixture(scope="module")
def anchor():
    with open(levels.REPO_ANCHOR_PATH) as f:
        return json.load(f)


def _ring(n_dcs: int) -> tuple[str, list[str]]:
    """YAML text of a ring member in the repo's topology format, and its origins."""
    ring = [f"{'edge' if p % 2 == 0 else 'micro'}_{p}" for p in range(n_dcs - 1)]

    def dc(name, dc_type, connect_to):
        pes, mips = TIERS[dc_type]
        text = (f"- !datacenter\n  name: {name}\n  type: {dc_type}\n  amount: 1\n"
                f"  hosts:\n    - !host\n      amount: 4\n      pes: {pes}\n      pe_mips: {mips}\n"
                f"      ram: 65536\n      storage: 4000000\n      bw: 100000\n"
                f"      vms:\n        - !vm\n          amount: 1\n          pes: {pes}\n"
                f"          pe_mips: {mips}\n          ram: 65536\n          size: 4000000\n"
                f"          bw: 100000\n")
        if connect_to:
            text += "  connect_to:\n" + "".join(f"    - {c}\n" for c in connect_to)
        return text

    text = dc("cloud", "cloud", None) + "".join(
        dc(name, name.split("_")[0], [ring[p - 1], ring[(p + 1) % len(ring)], "cloud"])
        for p, name in enumerate(ring)
    )
    return text, ring


def _capacity(n_dcs: int, hosts: int = 4) -> int:
    half = (n_dcs - 1) // 2
    return hosts * (TIERS["cloud"][0] + half * TIERS["edge"][0] + half * TIERS["micro"][0])


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory):
    """A small manifest laid out like common/topologies/ring/: <topologies>/ring/manifest.json."""
    ring_dir = tmp_path_factory.mktemp("topologies") / "ring"
    ring_dir.mkdir()
    members = []
    for member_id, n_dcs, load in [("R7", 7, {"rho": 1.0}), ("R11", 11, {"rho": 1.2}),
                                   ("R5-fixed", 5, {"lambda_of": "R7"})]:
        text, origins = _ring(n_dcs)
        (ring_dir / f"{member_id}.yml").write_text(text)
        members.append({
            "id": member_id, "yaml": f"ring/{member_id}.yml", "n_dcs": n_dcs, "origins": origins,
            "capacity_pes": _capacity(n_dcs), "tier_pe_share": {}, "load": load,
            "relabel_of": None, "permutation_seed": None, "isolates": "test",
        })
    path = ring_dir / "manifest.json"
    path.write_text(json.dumps({"members": members}))
    return str(path)


@pytest.fixture(scope="module")
def manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reports(manifest_path, anchor):
    return {r["member"]: r for r in levels.validate_manifest(manifest_path, anchor, VALIDATE_LEVELS)}


def _member(manifest, member_id):
    return next(m for m in manifest["members"] if m["id"] == member_id)


# ─── Anchor ─────────────────────────────────────────────────────────────────

def test_anchor_is_a_normalised_calibration(anchor):
    assert anchor["cores"]["values"] == [1, 2, 4, 8]
    assert math.isclose(sum(anchor["cores"]["probs"]), 1.0)
    assert set(anchor["sensitivity"]["probs"]) == set(levels.SENSITIVITIES)
    assert math.isclose(sum(anchor["sensitivity"]["probs"].values()), 1.0)
    shape = anchor["intensity"]["shape"]
    assert len(shape) == levels.SHAPE_BINS and min(shape) > 0
    assert math.isclose(np.mean(shape), 1.0, abs_tol=1e-6)
    assert anchor["derived"]["w_bar"] == pytest.approx(levels.w_bar(anchor))


# ─── Generation ─────────────────────────────────────────────────────────────

def test_level_jobs_follow_the_descriptor_contract(manifest, anchor):
    member = _member(manifest, "R11")
    jobs = levels.generate_level(member, 7, anchor, levels.resolve_lambda(manifest, "R11", anchor))
    assert list(jobs[0]) == ["jobId", "submissionDelay", "mi", "cores", "location",
                             "delaySensitivity", "deadline"]
    assert [j["jobId"] for j in jobs] == list(range(len(jobs)))
    arrivals = [j["submissionDelay"] for j in jobs]
    assert arrivals == sorted(arrivals)
    assert levels.FIRST_ARRIVAL <= arrivals[0] and arrivals[-1] <= levels.LAST_ARRIVAL
    for j in jobs:
        assert all(type(j[k]) is int for k in ("jobId", "submissionDelay", "mi", "cores", "deadline"))
        assert j["cores"] in (1, 2, 4, 8)
        assert j["location"] in member["origins"]
        runtime_ref, rest = divmod(j["mi"], levels.MIPS_REF)
        assert rest == 0 and levels.RUNTIME_MIN <= runtime_ref <= levels.RUNTIME_MAX
        lo, hi = levels.SLACK[j["delaySensitivity"]]
        assert lo <= j["deadline"] - math.ceil(j["mi"] / levels.DEADLINE_FLOOR_MIPS) <= hi


def test_a_level_is_a_pure_function_of_its_id(manifest, anchor):
    member, lam = _member(manifest, "R7"), 9.5
    assert levels.generate_level(member, 3, anchor, lam) == levels.generate_level(member, 3, anchor, lam)
    assert levels.generate_level(member, 3, anchor, lam) != levels.generate_level(member, 4, anchor, lam)
    assert levels.generate_level(member, 3, anchor, lam) != levels.generate_level(member, 3, anchor, lam, base_seed=1)


def test_members_sharing_a_lambda_share_the_job_stream(manifest, anchor):
    # lambda_of: same instance, different topology. Only location may differ.
    lam = levels.resolve_lambda(manifest, "R7", anchor)
    assert levels.resolve_lambda(manifest, "R5-fixed", anchor) == lam
    a = levels.generate_level(_member(manifest, "R7"), 11, anchor, lam)
    b = levels.generate_level(_member(manifest, "R5-fixed"), 11, anchor, lam)
    strip = lambda jobs: [{k: v for k, v in j.items() if k != "location"} for j in jobs]  # noqa: E731
    assert strip(a) == strip(b)


def test_generation_is_fast_at_the_largest_member_size(anchor):
    # C1-N19 (1088 PEs, rho 1.0) draws up to ~3500 jobs a level.
    member = {"origins": [f"o{i}" for i in range(18)]}
    lam = 3600 / (levels.LAST_ARRIVAL - levels.FIRST_ARRIVAL + 1)
    levels.generate_level(member, 0, anchor, lam)
    elapsed = []
    for level_id in range(20):
        start = time.perf_counter()
        levels.generate_level(member, level_id, anchor, lam)
        elapsed.append(time.perf_counter() - start)
    assert np.median(elapsed) < 0.050


# ─── Load ───────────────────────────────────────────────────────────────────

def test_w_bar_is_computed_from_the_anchor(anchor):
    # Closed form against Monte Carlo of the runtime rule.
    rng = np.random.default_rng(0)
    values, probs = np.array(anchor["cores"]["values"]), np.array(anchor["cores"]["probs"])
    cores = rng.choice(values, size=1_000_000, p=probs / probs.sum())
    runtime = np.clip(np.rint(rng.lognormal(levels._runtime_mu(cores), levels.RUNTIME_SIGMA)),
                      levels.RUNTIME_MIN, levels.RUNTIME_MAX)
    assert levels.w_bar(anchor) == pytest.approx((cores * runtime).mean(), rel=0.005)
    bigger = dict(anchor, cores={"values": [1, 2, 4, 8], "probs": [0, 0, 0, 1]})
    assert levels.w_bar(bigger) == pytest.approx(8 * levels.expected_runtime_ref(8))


def test_resolve_lambda_follows_the_manifest_load_rules(manifest, anchor):
    w = levels.w_bar(anchor)
    assert levels.resolve_lambda(manifest, "R7", anchor) == pytest.approx(1.0 * _capacity(7) / w)
    assert levels.resolve_lambda(manifest, "R11", anchor) == pytest.approx(1.2 * _capacity(11) / w)
    with pytest.raises(KeyError):
        levels.resolve_lambda(manifest, "missing", anchor)


# ─── Level-id splits ────────────────────────────────────────────────────────

def test_level_id_splits_are_the_literal_disjoint_ranges():
    assert levels.TRAIN_LEVELS == range(0, 100000)
    assert levels.VAL_LEVELS == range(1000000, 1000024)
    assert levels.TEST_LEVELS == range(2000000, 2000048)
    assert levels.LOCKBOX_LEVELS == range(3000000, 3000048)
    splits = [levels.TRAIN_LEVELS, levels.VAL_LEVELS, levels.TEST_LEVELS, levels.LOCKBOX_LEVELS]
    assert all(a.stop <= b.start for a, b in zip(splits, splits[1:]))


@pytest.mark.parametrize("split", ["val", "test", "lockbox"])
@pytest.mark.parametrize("num_workers", [1, 5, 16])
def test_eval_levels_partition_the_split_across_workers(split, num_workers):
    shards = [levels.eval_levels(split, rank, num_workers) for rank in range(num_workers)]
    assert sorted(i for shard in shards for i in shard) == list(levels.EVAL_SPLITS[split])


def test_eval_worker_takes_every_num_workers_th_id():
    assert levels.eval_levels("test", 1, 16) == [2000001, 2000017, 2000033]


def test_train_levels_are_sampled_reproducibly_inside_the_train_range():
    draws = [levels.sample_train_level(np.random.default_rng([0, rank])) for rank in range(16)]
    assert draws == [levels.sample_train_level(np.random.default_rng([0, rank])) for rank in range(16)]
    assert len(set(draws)) == 16 and all(d in levels.TRAIN_LEVELS for d in draws)


# ─── Validation ─────────────────────────────────────────────────────────────

def test_kendall_tau_b_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x = rng.integers(0, 12, size=5000)
    y = x + rng.integers(0, 9, size=5000)
    assert levels.kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-12)
    assert levels.kendall_tau_b(x, x) == pytest.approx(1.0)
    assert levels.kendall_tau_b(x, -x) == pytest.approx(-1.0)


def test_feasibility_check_catches_a_deadline_the_cloud_cannot_meet():
    cloud = {"type": "cloud", "hosts": [{"vms": [{"pes": 32, "pe_mips": 100}]}]}
    mi, cores = np.array([180, 180]), np.array([1, 1])
    # 3.0 network delay + 180/100 = 4.8 timesteps
    assert levels._feasible(cloud, cores, mi, np.array([5, 4])).tolist() == [True, False]
    micro = {"type": "micro", "hosts": [{"vms": [{"pes": 8, "pe_mips": 40}]}]}
    assert levels._feasible(micro, np.array([8, 16]), mi, np.array([5, 5])).tolist() == [True, False]


@pytest.mark.parametrize("member_id", ["R7", "R11", "R5-fixed"])
def test_validation_passes_every_check(reports, member_id):
    r = reports[member_id]
    assert abs(r["w_bar_rel_err"]) <= levels.W_BAR_TOLERANCE
    assert r["legal_degrees"] == [levels.LEGAL_DEGREE]
    assert r["infeasible_job_dc_pairs"] == 0
    assert abs(r["offered_load_rel_err"]) <= levels.OFFERED_LOAD_TOLERANCE
    assert r["top_site_ratio"] >= levels.HEAVY_TAIL_MIN_TOP_RATIO
    assert r["site_time_chi2_per_df"] >= levels.PHASE_CHI2_MIN
    assert r["failures"] == []


def test_site_phase_check_separates_stationary_from_shifted_origin_mixes():
    # The threshold applies to the mean over levels, as validate_member uses it.
    rng = np.random.default_rng(2)
    stationary, shifted = [], []
    for _ in range(200):
        arrivals = rng.integers(levels.FIRST_ARRIVAL, levels.LAST_ARRIVAL, endpoint=True, size=2000)
        location = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
        stationary.append(levels.site_time_chi2_per_df(location, arrivals))
        early = arrivals < (levels.FIRST_ARRIVAL + levels.LAST_ARRIVAL) / 2
        location = np.where(early & (rng.random(2000) < 0.1), "a", location)
        shifted.append(levels.site_time_chi2_per_df(location, arrivals))
    assert np.mean(stationary) < levels.PHASE_CHI2_MIN < np.mean(shifted)


def test_sensitivity_mix_is_the_design_mix(anchor):
    member = {"id": "m", "origins": ["a", "b", "c", "d"]}
    jobs = [j for level_id in range(20) for j in levels.generate_level(member, level_id, anchor, 10.0)]
    counts = {s: sum(j["delaySensitivity"] == s for j in jobs) for s in levels.SENSITIVITIES}
    for s, p in levels.SENSITIVITY_MIX.items():
        assert counts[s] / len(jobs) == pytest.approx(p, abs=0.02)


def test_validate_cli_reports_every_member_and_fails_on_a_failed_check(manifest_path):
    result = subprocess.run(
        [sys.executable, LEVELS_PY, "--validate", "--manifest", manifest_path, "--levels", "50"],
        capture_output=True, text=True,
    )
    for member_id in ("R7", "R11", "R5-fixed"):
        assert f"] {member_id}:" in result.stdout
    assert ("FAILED" in result.stdout) == (result.returncode == 1)


# ─── The real RING-N family (common/topologies/ring) ────────────────────────

@pytest.fixture(scope="module")
def ring_manifest():
    with open(RING_MANIFEST) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ring_reports(anchor):
    return {r["member"]: r for r in levels.validate_manifest(RING_MANIFEST, anchor, RING_VALIDATE_LEVELS)}


def test_ring_family_passes_every_check(ring_manifest, ring_reports):
    assert list(ring_reports) == [m["id"] for m in ring_manifest["members"]]
    for member_id, r in ring_reports.items():
        assert r["failures"] == [], (member_id, r["failures"])


@pytest.mark.parametrize("member_id, source_id", [("PI-S", "S"), ("PI-N19", "C1-N19")])
def test_relabelled_members_replay_their_source_job_stream(ring_manifest, anchor, member_id, source_id):
    # The relabelling lives in the YAML order only, so each level is the same instance.
    by_id = {m["id"]: m for m in ring_manifest["members"]}
    lam = levels.resolve_lambda(ring_manifest, source_id, anchor)
    assert levels.resolve_lambda(ring_manifest, member_id, anchor) == lam
    for level_id in levels.TEST_LEVELS[:5]:
        assert (levels.generate_level(by_id[member_id], level_id, anchor, lam)
                == levels.generate_level(by_id[source_id], level_id, anchor, lam))


def test_validate_cli_finds_the_anchor_in_the_container_layout(tmp_path):
    # common/docker-compose.yml mounts utils/, traces/ and topologies/ side by side under /mgr.
    # Copies, not symlinks: the OS resolves ".." through a symlink to the host layout.
    mgr = tmp_path / "mgr"
    (mgr / "utils").mkdir(parents=True)
    (mgr / "traces").mkdir()
    shutil.copy(LEVELS_PY, mgr / "utils")
    shutil.copy(levels.REPO_ANCHOR_PATH, mgr / "traces")
    shutil.copytree(os.path.dirname(RING_MANIFEST), mgr / "topologies" / "ring")
    result = subprocess.run(
        [sys.executable, str(mgr / "utils" / "levels.py"), "--validate",
         "--manifest", str(mgr / "topologies" / "ring" / "manifest.json"), "--levels", "5"],
        capture_output=True, text=True, cwd=mgr,
    )
    assert "Traceback" not in result.stderr, result.stderr
    assert "12 members pass" in result.stdout.splitlines()[-1]


# ─── Per-episode instances ──────────────────────────────────────────────────

def test_train_workers_draw_distinct_levels_and_eval_workers_partition_the_split():
    first = [levels.LevelSampler("train", rank, 16, seed=7).next() for rank in range(16)]
    assert len(set(first)) == 16
    assert all(level in levels.TRAIN_LEVELS for level in first)
    assert first == [levels.LevelSampler("train", rank, 16, seed=7).next() for rank in range(16)]
    assert first != [levels.LevelSampler("train", rank, 16, seed=8).next() for rank in range(16)]

    shares = [levels.eval_levels("val", rank, 16) for rank in range(16)]
    assert sorted(i for share in shares for i in share) == list(levels.VAL_LEVELS)
    sampler = levels.LevelSampler("val", 3, 16, seed=7)
    assert [sampler.next() for _ in range(4)] == shares[3] * 2


def test_level_source_emits_the_simulator_encoding(ring_manifest):
    topology = levels.load_topology(os.path.join(os.path.dirname(RING_MANIFEST), "S.yml"))
    names = [dc["name"] for dc in topology]
    source = levels.LevelSource(RING_MANIFEST, "S", names)
    jobs = json.loads(source.jobs_json(5))
    member = {m["id"]: m for m in ring_manifest["members"]}["S"]
    raw = levels.generate_level(member, 5, json.load(open(levels.REPO_ANCHOR_PATH)),
                                levels.resolve_lambda(ring_manifest, "S", json.load(open(levels.REPO_ANCHOR_PATH))))
    assert [j["location"] for j in jobs] == [names.index(j["location"]) for j in raw]
    assert [j["delaySensitivity"] for j in jobs] == [levels.SENSITIVITY_LEVELS[j["delaySensitivity"]] for j in raw]
    assert source.jobs_json(5) is source.jobs_json(5)  # cached
    with pytest.raises(ValueError, match="not in the topology"):
        levels.LevelSource(RING_MANIFEST, "C1-N19", names)

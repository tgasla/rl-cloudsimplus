"""RING-N topology family. Run: python3 -m pytest benchmark/tests/test_ring_family.py"""

import copy
import filecmp
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gen_ring_family as gen  # noqa: E402


def _member(member_id):
    with open(os.path.join(gen.RING_DIR, gen.MANIFEST_NAME)) as f:
        entry = next(e for e in json.load(f)["members"] if e["id"] == member_id)
    return entry, gen.load_topology(os.path.join(os.path.dirname(gen.RING_DIR), entry["yaml"]))


def _replace_member(member_id, **fields):
    names = ("id", "n_dcs", "edge", "micro", "load", "relabel_of", "seed", "isolates")
    return [tuple({**dict(zip(names, m)), **fields}.values()) if m[0] == member_id else m
            for m in gen.MEMBERS]


def test_committed_family_passes_the_audit():
    entries = gen.audit()
    assert [e["id"] for e in entries] == [m[0] for m in gen.MEMBERS]
    assert len(entries) == 12


def test_generation_is_deterministic_and_the_committed_files_are_current(tmp_path):
    out = tmp_path / "ring"
    gen.write_family(str(out))
    committed = sorted(os.listdir(gen.RING_DIR))
    assert sorted(os.listdir(out)) == committed
    _, mismatch, errors = filecmp.cmpfiles(gen.RING_DIR, out, committed, shallow=False)
    assert mismatch == [] and errors == [], f"regenerate the family: {mismatch or errors}"


def test_cloud_share_tracks_the_anchor_without_exceeding_it():
    entries = {e["id"]: e for e in gen.audit()}
    anchor = entries["S"]["tier_pe_share"]["cloud"]
    shares = [e["tier_pe_share"]["cloud"] for e in entries.values()]
    assert gen.cloud_hosts(gen.ANCHOR_N, gen.ANCHOR_HOSTS, gen.ANCHOR_HOSTS) == gen.ANCHOR_HOSTS
    assert all(s <= anchor for s in shares)
    assert anchor - min(shares) <= gen.MAX_CLOUD_SHARE_SPREAD


def test_each_arm_holds_its_controlled_factor():
    entries = {e["id"]: e for e in gen.audit()}
    members = {m[0]: m for m in gen.MEMBERS}
    for mid in ("C1-N7", "C1-N15", "C1-N19"):
        assert members[mid][2:4] == (gen.ANCHOR_HOSTS, gen.ANCHOR_HOSTS)   # per-DC size as S
        assert entries[mid]["load"] == {"rho": 1.0}
    c2 = {entries[mid]["capacity_pes"] for mid in ("C2-N7", "C2-N15", "C2-N19")}
    assert len(c2) == 1                                                     # identical within the arm
    assert abs(c2.pop() / entries["S"]["capacity_pes"] - 1) < 0.05          # ~S's capacity
    assert entries["GAM-lo"]["capacity_pes"] < entries["S"]["capacity_pes"] < entries["GAM-hi"]["capacity_pes"]
    for mid in ("C2-N7", "C2-N15", "C2-N19", "GAM-lo", "GAM-hi", "PI-S"):
        assert entries[mid]["load"] == {"lambda_of": "S"}
    assert entries["LOCK"]["capacity_pes"] < entries["S"]["capacity_pes"]
    assert entries["LOCK"]["load"]["rho"] > 1.0


@pytest.mark.parametrize("member_id, source_id", [("PI-S", "S"), ("PI-N19", "C1-N19")])
def test_relabelling_is_an_isomorphism_that_moves_action_indices(member_id, source_id):
    from utils import misc
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    def mask(dcs):
        indexed = misc._translate_connect_to_names_to_idx(copy.deepcopy(dcs))
        return JobPlacementEnv._build_location_mask(SimpleNamespace(max_datacenters=gen.MAX_DATACENTERS), indexed)

    (_, mine), (_, theirs) = _member(member_id), _member(source_id)
    names = [dc["name"] for dc in theirs]
    perm = np.array([names.index(dc["name"]) for dc in mine])      # my index i is source index perm[i]
    assert (perm != np.arange(len(perm))).any()
    m_mine, m_theirs = mask(mine), mask(theirs)
    assert not np.array_equal(m_mine, m_theirs)                        # the correct actions really move
    cols = np.concatenate([[0], perm + 1])                             # action = DC index + 1
    n = len(perm) + 1
    assert np.array_equal(m_mine[:, :n], m_theirs[perm][:, cols])     # ... and only by relabelling


def test_audit_rejects_a_connect_to_that_skips_a_ring_neighbour():
    entry, dcs = _member("S")
    dcs[1]["connect_to"][0] = dcs[5]["name"]
    with pytest.raises(AssertionError, match="ring neighbours"):
        gen.check_member(entry, dcs)


def test_audit_rejects_a_second_cloud():
    entry, dcs = _member("S")
    dcs[2]["type"] = "cloud"
    with pytest.raises(AssertionError, match="2 cloud DCs"):
        gen.check_member(entry, dcs)


def test_audit_rejects_an_origin_without_the_cloud_link():
    entry, dcs = _member("S")
    dcs[3]["connect_to"] = dcs[3]["connect_to"][:2]
    with pytest.raises(AssertionError, match="legal-action counts"):
        gen.check_legal_actions(entry["id"], dcs, entry["origins"])


def test_audit_rejects_the_spec_lockbox_isomorphic_to_c2_n15(tmp_path, monkeypatch):
    # The spec's LOCK (N=15, 3 hosts/DC) is a relabelling of C2-N15 once the cloud is corrected.
    monkeypatch.setattr(gen, "MEMBERS", _replace_member("LOCK", edge=3, micro=3))
    gen.write_family(str(tmp_path / "ring"))
    with pytest.raises(AssertionError, match="C2-N15 and LOCK are structurally identical"):
        gen.audit(str(tmp_path / "ring"))


def test_audit_rejects_a_manifest_that_breaks_the_shared_schema(tmp_path):
    ring = tmp_path / "ring"
    gen.write_family(str(ring))
    path = ring / gen.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["members"][0]["lambda"] = 9.8
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="S: manifest keys"):
        gen.audit(str(ring))


def test_audit_rejects_a_relabelling_that_changes_the_topology(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "MEMBERS", _replace_member("PI-S", micro=3))
    gen.write_family(str(tmp_path / "ring"))
    with pytest.raises(AssertionError, match="PI-S: not an exact relabelling of S"):
        gen.audit(str(tmp_path / "ring"))

"""benchmark/analyze.py on synthetic evaluation trees (no JVM).
Run: python3 -m pytest benchmark/tests/test_analyze.py"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze  # noqa: E402  (puts common/rl-manager on sys.path)
import evaluate  # noqa: E402
from utils.levels import EVAL_SPLITS, TEST_LEVELS, VAL_LEVELS  # noqa: E402
from utils.run_dir import write_run_status  # noqa: E402

LEVELS = list(TEST_LEVELS)
TARGETS = ["C1-N7", "C1-N15", "C1-N19", "C2-N7", "C2-N15", "C2-N19", "GAM-lo", "GAM-hi"]
MEMBERS = ["S"] + TARGETS + ["PI-S", "PI-N19"]
BUDGETS = analyze.FEW_SHOT_BUDGETS
REPS = 2000


# ─── Synthetic trees ────────────────────────────────────────────────────────

def write_references(path, members=MEMBERS, levels=LEVELS, g_rand=0.0, g_ref=1.0):
    """A references_summary file as run_references.py writes it; with the default floor 0 and
    reference 1, s equals G."""
    split = next(name for name, ids in EVAL_SPLITS.items() if levels[0] in ids)
    rows = [{"member": m, "split": split, "level_id": level, "G_rand": g_rand, "G_rand_sd": 0.01,
             "G_R1": 0.3, "G_R2": 0.2, "G_R3": 0.6, "G_R4_cdls": g_ref, "G_ref": g_ref,
             "ref_source": "R4", "zc_ceiling": 2.0, "rho_member": 1.0}
            for m in members for level in levels]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def write_evaluation(logs, member, returns, split="test", levels=LEVELS, **origin):
    """The evaluation.csv an evaluate run of the policy `origin` writes; `returns` is G on
    every level, or one G per level."""
    origin = {**dict.fromkeys(evaluate.PROVENANCE), "run_status": "completed", **origin}
    eval_dir = os.path.join(logs, "eval", origin["run"] or origin["arch"],
                            origin["checkpoint"].replace(">", "__"), member, split)
    os.makedirs(eval_dir)
    pd.DataFrame({"level_id": levels, "unshaped_return": returns, "member": member,
                  "split": split, "policy": "p", **origin}).to_csv(
        os.path.join(eval_dir, "evaluation.csv"), index=False)


def rollouts(k):
    """Env steps behind model_at_<k> with 2048-step rollouts."""
    return -(-k // 2048) * 2048


def source(arch, seed, checkpoint="best_val_model"):
    """A policy trained on S from scratch."""
    run = f"src/{arch}_{seed}"
    return dict(arch=arch, run=run, checkpoint=checkpoint, trained_on="S", source="S",
                source_run=run, seed=seed)


def tuned(arch, seed, member, k, finetune="full", start="best_val_model"):
    """model_at_<k> of a run fine-tuned on `member` from source(arch, seed, start)."""
    run = f"ft/{arch}_{seed}_{member}_{finetune}_{start}"
    return dict(arch=arch, run=run, checkpoint=f"{start}>model_at_{k}", trained_steps=rollouts(k),
                trained_on=member, finetune=finetune, source="S", source_run=f"src/{arch}_{seed}",
                seed=seed)


def scratch(arch, seed, member, k):
    """model_at_<k> of a run trained from scratch on `member`."""
    run = f"scratch/{arch}_{seed}_{member}"
    return dict(arch=arch, run=run, checkpoint=f"model_at_{k}", trained_steps=rollouts(k),
                trained_on=member, source=member, source_run=run, seed=seed)


def rule(name):
    return dict(arch=name, checkpoint="rule")


def tasks_of(logs, refs):
    references, split = analyze.load_references(refs)
    return analyze.task_scores(
        analyze.score_levels(analyze.load_evaluations(logs), references, split), split)


@pytest.fixture
def refs(tmp_path):
    return write_references(tmp_path / "references_summary_test.csv")


def test_analyze_reads_the_columns_evaluate_writes():
    assert set(analyze.REQUIRED) == set(evaluate.PROVENANCE) | {
        "member", "split", "level_id", "unshaped_return"}
    assert analyze.RULE == evaluate.RULE_CHECKPOINT


# ─── Normalisation and task scores ──────────────────────────────────────────

def test_scores_are_normalised_per_level_and_may_exceed_one(tmp_path):
    g_rand = np.linspace(-1.0, 0.0, len(LEVELS))
    g_ref = g_rand + np.linspace(1.0, 2.0, len(LEVELS))
    pd.DataFrame({"member": "S", "level_id": LEVELS, "G_rand": g_rand, "G_ref": g_ref}).to_csv(
        tmp_path / "refs.csv", index=False)
    s = np.where(np.arange(len(LEVELS)) % 2, 1.5, 0.25)          # s > 1 on half the levels
    write_evaluation(tmp_path / "logs", "S", g_rand + s * (g_ref - g_rand), **source("a5", 1))

    tasks = tasks_of(tmp_path / "logs", tmp_path / "refs.csv")
    assert tasks["score"].tolist() == pytest.approx([s.mean()])
    assert tasks[["block", "finetune", "levels"]].values.tolist() == [
        ["in_distribution", "none", 48]]


@pytest.mark.parametrize("change, error", [
    (lambda r: r.assign(G_ref=r["G_rand"]), "not above G_rand"),
    (lambda r: pd.concat([r, r.tail(1)]), "repeats 1"),
    (lambda r: r.assign(level_id=r["level_id"] - len(TEST_LEVELS)), "not those of one of"),
    (lambda r: r.drop(columns="G_ref"), "lacks columns"),
])
def test_unusable_references_are_refused(tmp_path, refs, change, error):
    change(pd.read_csv(refs)).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match=error):
        analyze.load_references(tmp_path / "bad.csv")


def test_the_references_split_comes_from_their_level_ids(tmp_path):
    path = write_references(tmp_path / "refs.csv", levels=list(VAL_LEVELS))
    assert analyze.load_references(path)[1] == "val"


def test_a_member_without_g_ref_on_every_level_is_not_scored(tmp_path, capsys):
    """As the reference runner leaves it before R4 has run: empty G_ref cells."""
    refs = pd.read_csv(write_references(tmp_path / "refs.csv", members=["S", "PI-S", "C1-N7"]))
    refs.loc[refs["member"] == "PI-S", "G_ref"] = np.nan
    refs.loc[(refs["member"] == "C1-N7") & (refs["level_id"] == LEVELS[5]), "G_rand"] = np.nan
    refs.to_csv(tmp_path / "refs.csv", index=False)
    for member in ("S", "PI-S", "C1-N7"):
        write_evaluation(tmp_path / "logs", member, 0.5, **source("a5", 1))

    tasks = tasks_of(tmp_path / "logs", tmp_path / "refs.csv")
    assert tasks["member"].tolist() == ["S"]
    out = capsys.readouterr().out
    assert "lack G_rand or G_ref on 48 of 48 levels of PI-S: PI-S is not scored" in out
    assert "lack G_rand or G_ref on 1 of 48 levels of C1-N7: C1-N7 is not scored" in out
    assert "evaluations on ['C1-N7', 'PI-S']: no references" in out


def test_references_without_any_complete_member_are_refused(tmp_path):
    refs = pd.read_csv(write_references(tmp_path / "refs.csv", members=["S", "PI-S"]))
    refs.assign(G_ref=np.nan, G_R4_cdls=np.nan, ref_source=np.nan).to_csv(
        tmp_path / "refs.csv", index=False)
    with pytest.raises(ValueError, match="no member has G_rand and G_ref on every level"):
        analyze.load_references(tmp_path / "refs.csv")


def test_what_cannot_be_scored_is_left_out_with_a_notice(tmp_path, capsys):
    logs, refs = tmp_path / "logs", write_references(tmp_path / "refs.csv", members=["S"])
    write_evaluation(logs, "S", 0.5, **source("a5", 1))
    write_evaluation(logs, "S", 0.5, split="val", levels=list(VAL_LEVELS), **source("a5", 1))
    write_evaluation(logs, "S", 0.5, **{**source("a5", 2), "run_status": "failed"})
    write_evaluation(logs, "LOCK", 0.5, **source("a5", 1))
    write_evaluation(logs / "_archive" / "old", "S", 0.9, **source("a5", 1))

    tasks = tasks_of(logs, refs)
    assert tasks[["run", "member", "score"]].values.tolist() == [["src/a5_1", "S", 0.5]]
    out = capsys.readouterr().out
    assert "ignored 24 rows on split(s) ['val']" in out
    assert "runs whose chain did not finish: src/a5_2" in out
    assert "evaluations on ['LOCK']: no references" in out


def test_references_missing_an_evaluated_level_are_refused(tmp_path):
    refs = write_references(tmp_path / "refs.csv", levels=LEVELS[:-1])
    write_evaluation(tmp_path / "logs", "S", 0.5, **source("a5", 1))
    with pytest.raises(ValueError, match="lack 1 evaluated"):
        tasks_of(tmp_path / "logs", refs)


def test_a_policy_evaluated_twice_on_a_member_is_refused(tmp_path, refs):
    write_evaluation(tmp_path / "logs", "S", 0.5, **source("a5", 1))
    write_evaluation(tmp_path / "logs" / "again", "S", 0.6, **source("a5", 1))
    with pytest.raises(ValueError, match="more than once"):
        tasks_of(tmp_path / "logs", refs)


def test_an_evaluation_without_every_level_is_refused(tmp_path, refs):
    write_evaluation(tmp_path / "logs", "S", 0.5, levels=LEVELS[:-1], **source("a5", 1))
    with pytest.raises(ValueError, match="exactly the 48 test levels"):
        tasks_of(tmp_path / "logs", refs)


def test_an_evaluation_without_provenance_is_refused(tmp_path):
    os.makedirs(tmp_path / "logs" / "old")
    pd.DataFrame({"level_id": LEVELS, "unshaped_return": 0.5, "member": "S", "split": "test"}) \
        .to_csv(tmp_path / "logs" / "old" / "evaluation.csv", index=False)
    with pytest.raises(ValueError, match="evaluate the policy again"):
        analyze.load_evaluations(tmp_path / "logs")


def test_every_run_kind_lands_in_its_block(tmp_path, refs):
    chain = {**tuned("a5", 1, "C1-N19", 5000), "run": "chain/2",
             "checkpoint": "best_val_model>final_model>model_at_5000"}
    cases = [                                                   # (member, policy, block)
        ("S", source("a5", 1), "in_distribution"),
        ("GAM-lo", source("a5", 1, "final_model"), "zero_shot"),
        ("PI-S", source("a5", 1), "relabel"),
        ("C1-N7", tuned("a5", 1, "C1-N7", 5000, finetune="head"), "few_shot"),
        ("GAM-hi", tuned("a5", 1, "GAM-hi", 50000, start="final_model"), "few_shot"),
        ("C1-N7", scratch("a5", 1, "C1-N7", 20000), "scratch"),
        ("S", rule("earliest-shortest-to-most-free-dc"), "in_distribution"),
        ("C1-N7", rule("earliest-shortest-to-most-free-dc"), "zero_shot"),
        ("C1-N19", chain, None),                                # two transfers
        ("S", tuned("a5", 1, "C1-N7", 20000), None),            # fine-tuned, evaluated elsewhere
        ("S", scratch("a5", 1, "C1-N7", 5000), None),           # zero-shot from another source
    ]
    for member, origin, _ in cases:
        write_evaluation(tmp_path / "logs", member, 0.5, **origin)

    found = {(t.arch, t.run if isinstance(t.run, str) else None, t.checkpoint, t.member):
             t.block if isinstance(t.block, str) else None
             for t in tasks_of(tmp_path / "logs", refs).itertuples()}
    assert found == {(o["arch"], o.get("run"), o["checkpoint"], m): b for m, o, b in cases}


# ─── Estimators ─────────────────────────────────────────────────────────────

def test_iqm_is_scipys_trimmed_mean():
    rng = np.random.default_rng(0)
    for shape in [(5, 4), (10, 9), (3, 1), (7, 3)]:
        scores = rng.exponential(size=shape)
        assert analyze.iqm(scores) == pytest.approx(stats.trim_mean(scores.ravel(), 0.25))
    batch = rng.exponential(size=(6, 5, 4))
    assert analyze.iqm(batch) == pytest.approx([stats.trim_mean(b.ravel(), 0.25) for b in batch])
    # 8 values: the 2 lowest and 2 highest go, the mean of 1, 2, 3, 4 stays
    assert analyze.iqm(np.array([[0.0, 100.0], [1.0, 2.0], [3.0, 4.0], [-50.0, 5.0]])) == 2.5


def test_iqm_of_symmetric_data_is_the_mean():
    for scores in (np.arange(24.0).reshape(6, 4), np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]]),
                   np.array([[-3.0, 0.0, 1.0], [5.0, 2.0, 1.0]])):    # about 1
        assert analyze.iqm(scores) == pytest.approx(scores.mean())


def test_the_stratified_bootstrap_ci_covers_the_true_iqm():
    """Four tasks whose means are symmetric about 0.5, so the equal-weight mixture over tasks
    and its IQM are too. The 95% CI must cover 0.5 in at least 90% of 200 experiments; a
    bootstrap that pooled runs across tasks would add the spread between the tasks and cover
    it nearly always."""
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(200):
        scores = rng.normal([0.2, 0.4, 0.6, 0.8], 0.15, size=(16, 4))
        _, lo, hi = analyze.stratified_bootstrap(scores, analyze.iqm, 1000, rng)
        covered += lo <= 0.5 <= hi
    assert 180 <= covered <= 197


def test_the_stratified_bootstrap_resamples_runs_within_each_task_only():
    """Cell (run i, task j) holds 10 j + i, so every resampled value names its run and task."""
    scores = np.array([[0.0, 10.0, 20.0], [1.0, 11.0, 21.0], [2.0, 12.0, 22.0], [3.0, 13.0, 23.0]])
    seen = []

    def recording_mean(values):
        seen.append(values)
        return analyze.mean(values)
    analyze.stratified_bootstrap(scores, recording_mean, 500, np.random.default_rng(0))
    replicates = next(v for v in seen if v.ndim == 3)
    assert replicates.shape == (500, 4, 3)
    for task in range(3):
        assert set(np.unique(replicates[..., task])) <= set(scores[:, task])
    assert len({r.tobytes() for r in replicates}) > 400        # and it does resample
    # each task draws its own runs (rliable's StratifiedBootstrap), not one draw of whole runs
    runs = replicates - 10.0 * np.arange(3)
    assert (runs[..., 0] != runs[..., 1]).mean() > 0.5


def test_tasks_whose_runs_agree_have_no_bootstrap_spread():
    """Only a resampling across tasks could move this IQM: every run agrees within its task."""
    scores = np.array([[0.1, 0.9]] * 5)
    assert analyze.stratified_bootstrap(scores, analyze.iqm, REPS, np.random.default_rng(0)) \
        == pytest.approx((0.5, 0.5, 0.5))


# ─── Aggregate ──────────────────────────────────────────────────────────────

def test_aggregate_is_the_iqm_over_runs_and_tasks_per_arch_checkpoint_and_block(tmp_path, refs):
    logs, rng, given = tmp_path / "logs", np.random.default_rng(3), {}
    for seed in (1, 2, 3):
        for member in ["S"] + TARGETS:
            for checkpoint in ("best_val_model", "final_model"):
                given[seed, member, checkpoint] = rng.uniform(-0.2, 1.3)
                write_evaluation(logs, member, given[seed, member, checkpoint],
                                 **source("a5", seed, checkpoint))
    for member in ["S"] + TARGETS:
        write_evaluation(logs, member, 0.4, **rule("earliest-most-critical-to-nearest-dc"))

    table = analyze.aggregate(tasks_of(logs, refs), reps=REPS).set_index(
        ["arch", "checkpoint", "block"])
    for checkpoint in ("best_val_model", "final_model"):
        row = table.loc[("a5", checkpoint, "zero_shot")]
        matrix = np.array([[given[s, m, checkpoint] for m in TARGETS] for s in (1, 2, 3)])
        assert row["iqm"] == pytest.approx(stats.trim_mean(matrix.ravel(), 0.25))
        assert row["mean"] == pytest.approx(matrix.mean())
        assert row["iqm_lo"] < row["iqm"] < row["iqm_hi"]
        assert (row["runs"], row["tasks"], row["finetune"]) == (3, 8, "none")
        assert row["members"].split(",") == sorted(TARGETS)
        in_distribution = [given[s, "S", checkpoint] for s in (1, 2, 3)]
        assert table.loc[("a5", checkpoint, "in_distribution"), "iqm"] == pytest.approx(
            stats.trim_mean(in_distribution, 0.25))
    rule_row = table.loc[("earliest-most-critical-to-nearest-dc", "rule", "zero_shot")]
    assert rule_row[["iqm", "iqm_lo", "iqm_hi", "runs", "tasks"]].tolist() == pytest.approx(
        [0.4, 0.4, 0.4, 1, 8])


def test_aggregate_ci_does_not_depend_on_the_other_rows(tmp_path, refs):
    logs = tmp_path / "logs"
    for seed in (1, 2, 3):
        write_evaluation(logs, "S", np.random.default_rng(seed).uniform(size=48),
                         **source("a5", seed))
    alone = analyze.aggregate(tasks_of(logs, refs), reps=REPS)
    write_evaluation(logs, "S", 0.1, **source("a1", 1))
    both = analyze.aggregate(tasks_of(logs, refs), reps=REPS)
    assert both[both["arch"] == "a5"].reset_index(drop=True).equals(alone)


def test_two_runs_of_one_arm_with_one_seed_are_refused(tmp_path, refs):
    """E.g. a permutation-augmented pretraining of a5 next to the plain one: the aggregate and
    the contrasts would pool them as replicates of one arm."""
    for member in ("S", "PI-S"):
        write_evaluation(tmp_path / "logs", member, 0.5, **source("a5", 1))
        write_evaluation(tmp_path / "logs", member, 0.6,
                         **{**source("a5", 1), "run": "perm/a5_1", "source_run": "perm/a5_1"})
    with pytest.raises(ValueError, match="share a seed(.|\n)*perm/a5_1"):
        tasks_of(tmp_path / "logs", refs)


def test_one_seed_in_different_blocks_is_no_clash(tmp_path, refs):
    """A source run and a scratch run on C1-N7 may share seed 1: zero_shot vs scratch."""
    write_evaluation(tmp_path / "logs", "C1-N7", 0.5, **source("a5", 1))
    write_evaluation(tmp_path / "logs", "C1-N7", 0.6,
                     **{**scratch("a5", 1, "C1-N7", 0), "checkpoint": "best_val_model",
                        "trained_steps": None})
    assert sorted(tasks_of(tmp_path / "logs", refs)["block"]) == ["scratch", "zero_shot"]


def test_a_group_with_unequal_runs_per_task_is_skipped_with_a_notice(tmp_path, refs, capsys):
    logs = tmp_path / "logs"
    for seed in (1, 2):
        write_evaluation(logs, "S", 0.5, **source("a5", seed))
        write_evaluation(logs, "C1-N7", 0.4, **source("a5", seed))
    write_evaluation(logs, "C1-N15", 0.3, **source("a5", 1))
    table = analyze.aggregate(tasks_of(logs, refs), reps=REPS)
    assert table["block"].tolist() == ["in_distribution"]
    out = capsys.readouterr().out
    assert "unequal runs per task {'C1-N15': 1, 'C1-N7': 2}" in out
    assert "block few_shot: no task scores, skipped" in out


# ─── Contrasts ──────────────────────────────────────────────────────────────

def test_contrasts_pair_every_source_run_with_itself(tmp_path, refs):
    """The runs sit at very different levels, but on each of them, under each checkpoint rule,
    every target scores exactly `shift` from S. Paired per run the contrast is exact (a CI of
    width 0); scores pooled across runs, or across checkpoint rules, would not give that."""
    logs = tmp_path / "logs"
    shifts = {"best_val_model": -0.1, "final_model": 0.2}
    for seed, level in ((1, 0.9), (2, 0.3), (3, 0.6)):
        for checkpoint, shift in shifts.items():
            write_evaluation(logs, "S", level + (seed == 1) * (checkpoint == "final_model"),
                             **source("a5", seed, checkpoint))
            for member in ("C1-N7", "C1-N15", "C1-N19", "PI-S"):
                write_evaluation(logs, member,
                                 level + (seed == 1) * (checkpoint == "final_model") + shift,
                                 **source("a5", seed, checkpoint))
    table = analyze.contrasts(tasks_of(logs, refs), reps=REPS).set_index(["checkpoint", "contrast"])
    for checkpoint, shift in shifts.items():
        for name in ("count_C1", "relabel"):
            row = table.loc[(checkpoint, name)]
            assert row[["mean", "lo", "hi"]].tolist() == pytest.approx([shift] * 3, abs=1e-12)
            assert row["runs"] == 3
    assert ("best_val_model", "count_C2") not in table.index           # no C2 evaluations


def test_a_run_missing_a_target_is_left_out_of_that_contrast_only(tmp_path, refs, capsys):
    logs = tmp_path / "logs"
    for seed in (1, 2):
        for member in ("S", "GAM-lo", "GAM-hi"):
            if (seed, member) != (2, "GAM-hi"):
                write_evaluation(logs, member, 0.5 + 0.1 * seed * (member != "S"),
                                 **source("a5", seed))
    table = analyze.contrasts(tasks_of(logs, refs), reps=REPS).set_index("contrast")
    assert table.loc["cap_lo", ["mean", "runs"]].tolist() == pytest.approx([0.15, 2])
    assert table.loc["cap_hi", ["mean", "runs"]].tolist() == pytest.approx([0.1, 1])
    assert "contrast cap_hi (a5, best_val_model): runs ['src/a5_2'] lack" in \
        capsys.readouterr().out


def test_direction_symmetry_and_the_difference_of_the_count_arms(tmp_path, refs):
    logs = tmp_path / "logs"
    given = {"S": 0.8, "C1-N7": 0.7, "C1-N15": 0.5, "C1-N19": 0.3, "C2-N7": 0.6, "C2-N15": 0.6,
             "C2-N19": 0.3, "PI-N19": 0.25}
    for seed, offset in ((1, 0.0), (2, 0.05)):
        for member, s in given.items():
            write_evaluation(logs, member, s + offset, **source("a5", seed))
    table = analyze.contrasts(tasks_of(logs, refs), reps=REPS).set_index("contrast")["mean"]
    assert table["direction_C1"] == pytest.approx(0.3 - 0.7)
    assert table["count_C1"] == pytest.approx(0.5 - 0.8)
    assert table["count_C2"] == pytest.approx(0.5 - 0.8)
    assert table["count_C1_minus_C2"] == pytest.approx(0.0, abs=1e-12)
    assert table["relabel_N19"] == pytest.approx(0.25 - 0.3)       # PI-N19 against C1-N19


# ─── Few-shot and transfer vs scratch ───────────────────────────────────────

def few_shot_campaign(logs, scratch_curve=(0.0, 0.1, 0.3, 0.5), seed=1, lift=0.0):
    """A run fine-tuned from S's best_val_model to C1-N7, whose FS_0 is S's zero-shot score
    there (0.2) and FS_5k/20k/50k 0.4/0.6/0.9 (+ lift), and a run trained from scratch on
    C1-N7 with model_at_<k> at every budget."""
    write_evaluation(logs, "C1-N7", 0.2 + lift, **source("a5", seed))
    for k, s in zip(BUDGETS[1:], (0.4, 0.6, 0.9)):
        write_evaluation(logs, "C1-N7", s + lift, **tuned("a5", seed, "C1-N7", k))
    for k, s in zip(BUDGETS, scratch_curve):
        write_evaluation(logs, "C1-N7", s, **scratch("a5", seed, "C1-N7", k))


def test_few_shot_curves_and_aulc_by_hand(tmp_path, refs):
    few_shot_campaign(tmp_path / "logs")
    curves = analyze.few_shot(tasks_of(tmp_path / "logs", refs)).set_index("finetune")
    run, base = curves.loc["full"], curves.loc["scratch"]
    assert [run[f"fs_{k}"] for k in BUDGETS] == pytest.approx([0.2, 0.4, 0.6, 0.9])
    assert [run[f"steps_{k}"] for k in BUDGETS] == [0, 6144, 20480, 51200]
    # (0.2+0.4)/2*5000 + (0.4+0.6)/2*15000 + (0.6+0.9)/2*30000 = 31500, over 50000
    assert run["aulc"] == pytest.approx(0.63)
    assert run["checkpoint"] == "best_val_model>model_at_k"
    # (0+0.1)/2*5000 + (0.1+0.3)/2*15000 + (0.3+0.5)/2*30000 = 15250, over 50000
    assert base["aulc"] == pytest.approx(0.305)
    assert base["checkpoint"] == "model_at_k"


def test_a_fine_tuned_runs_own_model_at_0_comes_before_the_sources_score(tmp_path, refs):
    few_shot_campaign(tmp_path / "logs")
    write_evaluation(tmp_path / "logs", "C1-N7", 0.25, **tuned("a5", 1, "C1-N7", 0))
    run = analyze.few_shot(tasks_of(tmp_path / "logs", refs)).set_index("finetune").loc["full"]
    assert (run["fs_0"], run["steps_0"]) == pytest.approx((0.25, 0))


def test_a_checkpoint_between_budgets_is_named_not_dropped_silently(tmp_path, refs, capsys):
    few_shot_campaign(tmp_path / "logs")
    write_evaluation(tmp_path / "logs", "C1-N7", 0.5, **tuned("a5", 1, "C1-N7", 10000))
    tasks = tasks_of(tmp_path / "logs", refs)
    run = analyze.few_shot(tasks).set_index("finetune").loc["full"]
    assert run["aulc"] == pytest.approx(0.63)                  # the budgets' curve is unchanged
    assert "k = [10000] are not budgets (0, 5000, 20000, 50000)" in capsys.readouterr().out
    assert "best_val_model>model_at_10000" in set(tasks["checkpoint"])


def test_a_curve_missing_a_budget_has_no_aulc(tmp_path, refs, capsys):
    write_evaluation(tmp_path / "logs", "C1-N7", 0.4, **tuned("a5", 1, "C1-N7", 5000))
    curve = analyze.few_shot(tasks_of(tmp_path / "logs", refs)).iloc[0]
    assert np.isnan(curve["fs_0"]) and np.isnan(curve["fs_20000"]) and np.isnan(curve["aulc"])
    assert "1 curves miss one of the budgets" in capsys.readouterr().out


def test_transfer_vs_scratch_and_the_area_ratio_by_hand(tmp_path, refs):
    few_shot_campaign(tmp_path / "logs")
    row = analyze.transfer_vs_scratch(
        analyze.few_shot(tasks_of(tmp_path / "logs", refs))).iloc[0]
    assert [row[f"gain_{k}"] for k in BUDGETS] == pytest.approx([0.2, 0.3, 0.3, 0.4])
    assert (row["aulc_transfer"], row["aulc_scratch"]) == pytest.approx((0.63, 0.305))
    assert row["area_ratio"] == pytest.approx((0.63 - 0.305) / 0.305)       # 1.0656
    assert row["checkpoint"] == "best_val_model>model_at_k vs model_at_k"
    assert (row["runs_transfer"], row["runs_scratch"]) == (1, 1)


def test_transfer_vs_scratch_averages_each_arms_runs(tmp_path, refs):
    few_shot_campaign(tmp_path / "logs", seed=1)
    few_shot_campaign(tmp_path / "logs", seed=2, lift=0.1, scratch_curve=(0.1, 0.2, 0.4, 0.6))
    row = analyze.transfer_vs_scratch(
        analyze.few_shot(tasks_of(tmp_path / "logs", refs))).iloc[0]
    assert (row["aulc_transfer"], row["aulc_scratch"]) == pytest.approx((0.68, 0.355))
    assert row["area_ratio"] == pytest.approx((0.68 - 0.355) / 0.355)
    assert (row["runs_transfer"], row["runs_scratch"]) == (2, 2)


def test_no_area_ratio_against_a_scratch_curve_at_or_below_the_floor(tmp_path, refs, capsys):
    few_shot_campaign(tmp_path / "logs", scratch_curve=(-0.4, -0.2, 0.0, 0.1))
    row = analyze.transfer_vs_scratch(
        analyze.few_shot(tasks_of(tmp_path / "logs", refs))).iloc[0]
    # -0.3*5000 - 0.1*15000 + 0.05*30000 = -1500, over 50000
    assert row["aulc_scratch"] == pytest.approx(-0.03) and np.isnan(row["area_ratio"])
    assert "area ratio left empty" in capsys.readouterr().out


# ─── Output ─────────────────────────────────────────────────────────────────

def test_check_tables_wants_a_checkpoint_rule_in_every_row():
    good = pd.DataFrame({"checkpoint": ["best_val_model"], "iqm": [0.5]})
    analyze.check_tables({"good": good})
    with pytest.raises(ValueError, match="table bad has no checkpoint column"):
        analyze.check_tables({"good": good, "bad": good.drop(columns="checkpoint")})
    for blank in (None, np.nan, "", " "):
        with pytest.raises(ValueError, match="1 rows name no checkpoint rule"):
            analyze.check_tables({"t": pd.DataFrame({"checkpoint": ["final_model", blank]})})


def test_nothing_is_written_when_a_table_fails_the_check(tmp_path):
    with pytest.raises(ValueError):
        analyze.write_tables({"a": pd.DataFrame({"checkpoint": ["rule"]}),
                              "b": pd.DataFrame({"x": [1]})}, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def full_campaign(logs):
    """Every run kind: two archs x 3 seeds trained on S (best_val_model and final_model on
    every member), full/head/extractor fine-tuning to two targets at every budget, scratch runs
    on those targets, a rule-based policy everywhere and a chain of two transfers."""
    rng = np.random.default_rng(7)
    for arch in ("a1", "a5"):
        for seed in (1, 2, 3):
            for checkpoint in ("best_val_model", "final_model"):
                for member in MEMBERS:
                    write_evaluation(logs, member, rng.uniform(0, 1, 48),
                                     **source(arch, seed, checkpoint))
            for member in ("C1-N7", "GAM-lo"):
                for finetune in ("full", "head", "extractor"):
                    for k in BUDGETS[1:]:
                        write_evaluation(logs, member, rng.uniform(0, 1, 48),
                                         **tuned(arch, seed, member, k, finetune))
                for k in BUDGETS:
                    write_evaluation(logs, member, rng.uniform(0.1, 1, 48),
                                     **scratch(arch, seed, member, k))
    for member in MEMBERS:
        write_evaluation(logs, member, rng.uniform(0, 1, 48),
                         **rule("earliest-shortest-to-most-free-dc"))
    chain = {**tuned("a5", 1, "C1-N19", 5000), "run": "chain/2",
             "checkpoint": "best_val_model>final_model>model_at_5000"}
    write_evaluation(logs, "C1-N19", 0.5, **chain)


def test_main_writes_every_table_with_its_checkpoint_rule(tmp_path, refs, capsys):
    full_campaign(tmp_path / "logs")
    out = tmp_path / "out"
    analyze.main(["--logs", str(tmp_path / "logs"), "--references", refs, "--out", str(out),
                  "--reps", "500"])
    tables = {name[:-4]: pd.read_csv(out / name) for name in os.listdir(out)}
    assert sorted(tables) == ["aggregate", "contrasts", "few_shot", "task_scores",
                              "transfer_vs_scratch"]
    for table in tables.values():
        assert table["checkpoint"].notna().all()
    assert len(tables["contrasts"]) == 2 * 2 * len(analyze.CONTRASTS)
    assert len(tables["few_shot"]) == 2 * 3 * 2 * 3 + 2 * 3 * 2          # fine-tuned + scratch
    assert tables["few_shot"]["aulc"].notna().all()
    assert len(tables["transfer_vs_scratch"]) == 2 * 2 * 3
    blocks = tables["aggregate"].groupby("block").size().to_dict()
    assert blocks == {"in_distribution": 2 * 2 + 1, "zero_shot": 2 * 2 + 1, "relabel": 2 * 2 + 1,
                      "few_shot": 2 * 3 * 3, "scratch": 2 * 4}
    printed = capsys.readouterr().out
    assert "block lockbox: no task scores, skipped" in printed
    assert "1 task scores are in no block" in printed


def test_main_writes_what_it_can_and_says_what_it_skipped(tmp_path, refs, capsys):
    logs = tmp_path / "logs"
    for seed in (1, 2):
        for member in ("S", "C1-N7", "C1-N15", "C1-N19"):
            write_evaluation(logs, member, 0.5, **source("a5", seed))
    analyze.main(["--logs", str(logs), "--references", refs, "--out", str(tmp_path / "out"),
                  "--reps", "200"])
    assert sorted(os.listdir(tmp_path / "out")) == ["aggregate.csv", "contrasts.csv",
                                                    "task_scores.csv"]
    printed = capsys.readouterr().out
    for skipped in ("few_shot: no model_at_<k> evaluations", "transfer_vs_scratch: no complete",
                    "block few_shot: no task scores", "a5: no ['final_model'] evaluations",
                    "contrast count_C2 skipped for a5/best_val_model: no run evaluated on all of "
                    "['C2-N7', 'C2-N15', 'C2-N19', 'S']"):
        assert skipped in printed


def test_a_table_with_nothing_to_report_now_leaves_no_stale_copy(tmp_path, refs, capsys):
    logs, out = tmp_path / "logs", tmp_path / "out"
    few_shot_campaign(logs)
    analyze.main(["--logs", str(logs), "--references", refs, "--out", str(out), "--reps", "200"])
    assert (out / "few_shot.csv").exists() and (out / "transfer_vs_scratch.csv").exists()
    (out / "notes.txt").write_text("not a table")

    fresh = tmp_path / "fresh"
    for member in ("S", "C1-N7", "C1-N15", "C1-N19"):
        write_evaluation(fresh, member, 0.5, **source("a5", 1))
    capsys.readouterr()
    analyze.main(["--logs", str(fresh), "--references", refs, "--out", str(out), "--reps", "200"])
    assert sorted(os.listdir(out)) == ["aggregate.csv", "contrasts.csv", "notes.txt",
                                       "task_scores.csv"]
    printed = capsys.readouterr().out
    assert f"removed {out / 'few_shot.csv'}, written by an earlier call" in printed
    assert f"removed {out / 'transfer_vs_scratch.csv'}, written by an earlier call" in printed


def test_nothing_is_written_when_no_evaluation_can_be_scored(tmp_path, refs, capsys):
    """E.g. --references of the test split for evaluations on val: the tables of an earlier
    call stay as they are."""
    logs, out = tmp_path / "logs", tmp_path / "out"
    write_evaluation(logs, "S", 0.5, split="val", levels=list(VAL_LEVELS), **source("a5", 1))
    out.mkdir()
    (out / "aggregate.csv").write_text("checkpoint\nbest_val_model\n")
    with pytest.raises(ValueError, match="can be scored"):
        analyze.main(["--logs", str(logs), "--references", refs, "--out", str(out)])
    assert os.listdir(out) == ["aggregate.csv"]
    assert "ignored 24 rows on split(s) ['val']" in capsys.readouterr().out


def test_end_to_end_from_run_directories(tmp_path, monkeypatch, refs):
    """run_status.json files as the entrypoint writes them and evaluate.evaluate() with the
    simulator stubbed (a policy's return on every level is the score given here), then main."""
    logs, out = tmp_path / "logs", tmp_path / "out"

    def run(run_dir, budgets=(), **params):
        os.makedirs(logs / run_dir)
        write_run_status(str(logs / run_dir), {"feature_extractor": "a5", "seed": 1,
                                               "cloudlet_to_dc_mapping": "rl", **params},
                         "completed")
        for k in budgets:
            with open(logs / run_dir / f"model_at_{k}.json", "w") as f:
                json.dump({"k": k, "trained_steps": rollouts(k)}, f)

    run("src/a5", mode="train", benchmark_member="S")
    run("ft/head", BUDGETS[1:], mode="transfer", benchmark_member="C1-N7",
        train_model_dir="src/a5", finetune="head")
    run("scratch/c1n7", BUDGETS, mode="train", benchmark_member="C1-N7")
    run("chain/c1n19", mode="transfer", benchmark_member="C1-N19", train_model_dir="ft/head",
        checkpoint="final_model")
    given = {("src/a5", "best_val_model", m): s for m, s in
             (("S", 0.9), ("C1-N7", 0.2), ("C1-N15", 0.5), ("C1-N19", 0.5))}
    given.update({("ft/head", f"model_at_{k}", "C1-N7"): s
                  for k, s in zip(BUDGETS[1:], (0.4, 0.6, 0.9))})
    given.update({("scratch/c1n7", f"model_at_{k}", "C1-N7"): s
                  for k, s in zip(BUDGETS, (0.0, 0.1, 0.3, 0.5))})
    given[("chain/c1n19", "best_val_model", "C1-N19")] = 0.7
    given[(None, "earliest-most-critical-to-nearest-dc", "C1-N7")] = 0.3

    class Env:
        def __init__(self, params):
            self.params = params

        def close(self):
            pass

    class Algorithm:
        @staticmethod
        def load(path, env, device):
            return None

    def play(env, predict, quotas):
        p = env.params
        rl = p["cloudlet_to_dc_mapping"] == "rl"
        policy = (p["train_model_dir"], p["checkpoint"]) if rl else \
            (None, p["cloudlet_to_dc_mapping"])
        return [{"level_id": level, "unshaped_return": given[(*policy, p["benchmark_member"])]}
                for level in LEVELS]
    monkeypatch.setattr(evaluate, "vectorize_env", lambda env, algorithm, **k: Env(k["params"]))
    monkeypatch.setattr(evaluate, "get_algorithm", lambda name, params: Algorithm)
    monkeypatch.setattr(evaluate, "get_suitable_device", lambda name: "cpu")
    monkeypatch.setattr(evaluate, "play_levels", play)
    for i, (run_dir, checkpoint, member) in enumerate(given):
        os.makedirs(logs / "eval" / str(i))
        policy = {"cloudlet_to_dc_mapping": "rl", "train_model_dir": run_dir,
                  "checkpoint": checkpoint} if run_dir else {"cloudlet_to_dc_mapping": checkpoint}
        evaluate.evaluate({"base_log_dir": str(logs), "level_split": "test", "num_cpu": 4,
                           "benchmark_member": member, "rl_algorithm": "MaskablePPO",
                           "max_jobs_waiting": 32, "log_dir": str(logs / "eval" / str(i)),
                           **policy}, [])

    analyze.main(["--logs", str(logs), "--references", refs, "--out", str(out), "--reps", "200"])
    curves = pd.read_csv(out / "few_shot.csv").set_index("finetune")
    assert curves.loc["head", "checkpoint"] == "best_val_model>model_at_k"
    assert [curves.loc["head", f"steps_{k}"] for k in BUDGETS] == [0, 6144, 20480, 51200]
    assert curves.loc["head", "aulc"] == pytest.approx(0.63)
    assert curves.loc["scratch", "aulc"] == pytest.approx(0.305)
    versus = pd.read_csv(out / "transfer_vs_scratch.csv").iloc[0]
    assert versus["area_ratio"] == pytest.approx((0.63 - 0.305) / 0.305)
    contrasts = pd.read_csv(out / "contrasts.csv").set_index("contrast")
    assert contrasts.loc["count_C1", "mean"] == pytest.approx((0.2 + 0.5 + 0.5) / 3 - 0.9)
    assert contrasts.loc["direction_C1", "mean"] == pytest.approx(0.5 - 0.2)
    tasks = pd.read_csv(out / "task_scores.csv").set_index("run")
    assert tasks.loc["chain/c1n19", ["checkpoint", "source", "trained_on"]].tolist() == [
        "best_val_model>final_model>best_val_model", "S", "C1-N19"]
    assert pd.isna(tasks.loc["chain/c1n19", "block"])
    rule_row = tasks[tasks["arch"] == "earliest-most-critical-to-nearest-dc"].iloc[0]
    assert (rule_row["checkpoint"], rule_row["block"]) == ("rule", "zero_shot")
    assert rule_row["score"] == pytest.approx(0.3)

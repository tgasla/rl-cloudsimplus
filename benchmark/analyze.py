"""The paper's numbers from evaluation.csv files (mode: evaluate), one row per (policy, level).

    python3 benchmark/analyze.py --logs common/logs \\
        --references benchmark/results/references_summary_test.csv --out <dir>

Every row carries the unshaped return G and its policy's provenance (provenance() in
common/rl-manager/evaluate.py). `checkpoint` names the rule that produced the policy, from the
source run to the evaluated one: best_val_model, final_model, best_val_model>model_at_5000 (the
model_at_5000 checkpoint of a run fine-tuned from the source's best_val_model), ..., or "rule".

  score       s = (G - G_rand) / (G_ref - G_rand) per (member, level), with G_rand (random-
              feasible floor R0, mean of 8 rollouts) and G_ref (clairvoyant reference) from the
              reference runner's references_summary_<split>.csv. s > 1 is a legitimate result:
              the policy beat the clairvoyant reference. A member whose references lack G_rand
              or G_ref on any level is not scored (with a notice).
  task score  mean s over a member's levels for one policy: ZS(m, S->T) for a policy trained on
              S, FS_k(m, S->T) for the checkpoint after k fine-tuning steps on T.
  aggregate   IQM (25% trimmed mean) over runs x tasks per (arch, finetune, checkpoint, block),
              with a 95% percentile CI from a stratified bootstrap that resamples runs within
              each task (the estimators of rliable, Agarwal et al. 2021).
  contrasts   Delta = ZS(S->T) - ZS(S->S) per source run, both scores from the same run and
              checkpoint, averaged over the contrast's targets; the mean over runs with a
              percentile bootstrap CI that resamples runs, which keeps each run's pairing.
  few-shot    FS_k at k in FEW_SHOT_BUDGETS (FS_0: the policy fine-tuning started from) and
              AULC, the trapezoid over k divided by the largest k.
  transfer    FS_k(fine-tuned from S) - FS_k(trained from scratch on T), and Taylor & Stone's
              (2009) area ratio (A_transfer - A_scratch) / A_scratch.

Every table names, in every row, the checkpoint rule its numbers come from (check_tables runs
before anything is written), and no reported number takes a maximum over a learning curve
(benchmark/lint_analysis.py enforces both on this file). A table with nothing to report is not
written, and a copy left in --out by an earlier call is removed, both with a notice.
"""
import argparse
import glob
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import integrate, stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "common", "rl-manager"))

from utils.levels import EVAL_SPLITS  # noqa: E402
from utils.run_dir import ARCHIVE_DIRNAME  # noqa: E402

SOURCE = "S"
RULE = "rule"                       # evaluate.py's checkpoint for a rule-based policy
# The task set a member belongs to for a policy trained on SOURCE (and for a rule-based policy).
BLOCKS = {
    SOURCE: "in_distribution",
    **dict.fromkeys(["C1-N7", "C1-N15", "C1-N19", "C2-N7", "C2-N15", "C2-N19", "GAM-lo", "GAM-hi"],
                    "zero_shot"),
    "PI-S": "relabel", "PI-N19": "relabel",
    "LOCK": "lockbox",
}
BLOCK_NAMES = ("in_distribution", "zero_shot", "relabel", "lockbox", "few_shot", "scratch")
# name -> (targets, baselines): per source run, mean ZS over targets - mean ZS over baselines
C1, C2 = ["C1-N7", "C1-N15", "C1-N19"], ["C2-N7", "C2-N15", "C2-N19"]
CONTRASTS = {
    "count_C1": (C1, [SOURCE]),
    "count_C2": (C2, [SOURCE]),
    "cap_lo": (["GAM-lo"], [SOURCE]),
    "cap_hi": (["GAM-hi"], [SOURCE]),
    "relabel": (["PI-S"], [SOURCE]),
    "relabel_N19": (["PI-N19"], ["C1-N19"]),    # the second relabelling control, against C1-N19
    "count_C1_minus_C2": (C1, C2),              # H1: count_C1 - count_C2, S cancels
    "direction_C1": (["C1-N19"], ["C1-N7"]),    # direction symmetry
}
FEW_SHOT_BUDGETS = (0, 5000, 20000, 50000)      # ascending; AULC divides by the last
REPS = 50_000
TABLES = ("task_scores", "aggregate", "contrasts", "few_shot", "transfer_vs_scratch")

REQUIRED = ["member", "split", "level_id", "unshaped_return", "arch", "run", "checkpoint",
            "trained_steps", "trained_on", "finetune", "source", "source_run", "seed",
            "run_status"]
POLICY = ["arch", "run", "checkpoint"]
LINEAGE = ["seed", "source", "source_run", "trained_on", "finetune", "trained_steps"]
TASK_COLUMNS = ["arch", "finetune", "checkpoint", "block", "member", "score", "mean_return",
                "levels", "seed", "run", "source", "source_run", "trained_on", "trained_steps"]


def notice(message: str) -> None:
    print(f"[analyze] {message}")


# ─── Loading and normalisation ──────────────────────────────────────────────

def load_evaluations(logs_root: str) -> pd.DataFrame:
    """Every evaluation.csv under logs_root, except archived runs, with the eval_dir it is in."""
    frames = []
    for path in sorted(glob.glob(os.path.join(logs_root, "**", "evaluation.csv"), recursive=True)):
        eval_dir = os.path.relpath(os.path.dirname(path), logs_root)
        if eval_dir.split(os.sep)[0] == ARCHIVE_DIRNAME:
            continue
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks {missing}: evaluated before evaluate.py recorded "
                             f"provenance; evaluate the policy again")
        frames.append(frame.assign(eval_dir=eval_dir))
    if not frames:
        raise FileNotFoundError(f"no evaluation.csv under {logs_root}")
    return pd.concat(frames, ignore_index=True).astype({"seed": "Int64", "trained_steps": "Int64"})


def load_references(path: str) -> tuple[pd.DataFrame, str]:
    """[member, level_id, G_rand, G_ref] from the reference runner's summary, and the split
    its level ids come from. A member lacking G_rand or G_ref on some level (R4 not run yet)
    is left out with a notice."""
    refs = pd.read_csv(path)
    missing = sorted({"member", "level_id", "G_rand", "G_ref"} - set(refs.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    refs = refs[["member", "level_id", "G_rand", "G_ref"]]
    repeated = refs.duplicated(["member", "level_id"])
    if repeated.any():
        raise ValueError(f"{path} repeats {repeated.sum()} (member, level_id) pairs")
    splits = [name for name, ids in EVAL_SPLITS.items() if refs["level_id"].isin(ids).all()]
    if len(splits) != 1:
        raise ValueError(f"{path}: the level ids are not those of one of {list(EVAL_SPLITS)}")
    unset = refs[refs[["G_rand", "G_ref"]].isna().any(axis=1)].groupby("member").size()
    for member, n in unset.items():
        notice(f"references lack G_rand or G_ref on {n} of {(refs['member'] == member).sum()} "
               f"levels of {member}: {member} is not scored")
    refs = refs[~refs["member"].isin(unset.index)]
    if refs.empty:
        raise ValueError(f"{path}: no member has G_rand and G_ref on every level")
    inverted = refs[~(refs["G_ref"] > refs["G_rand"])]
    if len(inverted):
        raise ValueError(f"{path}: G_ref is not above G_rand for {len(inverted)} (member, "
                         f"level) pairs, so s is undefined there:\n{inverted}")
    return refs, splits[0]


def score_levels(evals: pd.DataFrame, refs: pd.DataFrame, split: str) -> pd.DataFrame:
    """The evaluations of finished policies on `split`, with s per level."""
    elsewhere = evals["split"] != split
    if elsewhere.any():
        notice(f"ignored {elsewhere.sum()} rows on split(s) "
               f"{sorted(evals.loc[elsewhere, 'split'].unique())}: the references are for {split}")
    evals = evals[~elsewhere]
    unfinished = evals["run_status"].notna() & (evals["run_status"] != "completed")
    if unfinished.any():
        notice("ignored the evaluations of runs whose chain did not finish: "
               + ", ".join(sorted(evals.loc[unfinished, "run"].unique())))
    evals = evals[~unfinished]
    unreferenced = ~evals["member"].isin(refs["member"])
    if unreferenced.any():
        notice(f"ignored the evaluations on {sorted(evals.loc[unreferenced, 'member'].unique())}:"
               f" no references for them")
    scored = evals[~unreferenced].merge(refs, on=["member", "level_id"], how="left",
                                         validate="many_to_one")
    if scored["G_ref"].isna().any():
        missing = scored.loc[scored["G_ref"].isna(), ["member", "level_id"]].drop_duplicates()
        raise ValueError(f"the references lack {len(missing)} evaluated (member, level_id) "
                         f"pairs:\n{missing}")
    return scored.assign(
        score=(scored["unshaped_return"] - scored["G_rand"]) / (scored["G_ref"] - scored["G_rand"]))


def block_of(task) -> str | None:
    """The task set a (policy, member) score is aggregated in; None: task_scores.csv only."""
    depth = task.checkpoint.count(">")
    if task.checkpoint == RULE or (depth == 0 and task.source == SOURCE):
        return BLOCKS.get(task.member)
    if depth == 0 and task.member == task.source:
        return "scratch"
    if depth == 1 and task.source == SOURCE and task.member == task.trained_on:
        return "few_shot"
    return None


def task_scores(scored: pd.DataFrame, split: str) -> pd.DataFrame:
    """Mean s per (policy, member) over exactly the split's levels, with its block. Within a
    block every seed is one run: the tables treat the runs of an (arch, finetune, checkpoint)
    as replicates of one arm, so two runs with one seed there are two arms they cannot tell
    apart (e.g. two pretraining variants of one feature_extractor)."""
    key = POLICY + ["member"]
    repeated = scored[scored.duplicated(key + ["level_id"], keep=False)]
    if len(repeated):
        raise ValueError("policies evaluated more than once on a member (remove or archive all "
                         "but one):\n" + repeated.groupby(key, dropna=False)["eval_dir"]
                         .unique().to_string())
    levels = sorted(EVAL_SPLITS[split])
    grouped = scored.groupby(key + LINEAGE, dropna=False)
    partial = grouped["level_id"].apply(lambda ids: sorted(ids) != levels)
    if partial.any():
        raise ValueError(f"evaluations without exactly the {len(levels)} {split} levels:\n"
                         f"{partial[partial]}")
    tasks = grouped.agg(score=("score", "mean"), mean_return=("unshaped_return", "mean"),
                        levels=("level_id", "size")).reset_index()
    tasks["finetune"] = tasks["finetune"].fillna("none")
    tasks["block"] = [block_of(task) for task in tasks.itertuples()]
    arm = ["arch", "finetune", "checkpoint", "block", "member", "seed"]
    in_block = tasks[tasks["block"].notna() & tasks["seed"].notna()]
    shared = in_block[in_block.duplicated(arm, keep=False)]
    if len(shared):
        raise ValueError("runs of one arm share a seed, so they are not replicates of it; "
                         "archive all but one, or analyze the arms from separate --logs roots:\n"
                         + shared[arm + ["run"]].to_string(index=False))
    return tasks[TASK_COLUMNS]


# ─── Estimators ─────────────────────────────────────────────────────────────

def iqm(scores: np.ndarray) -> np.ndarray:
    """Interquartile mean over the last two axes (runs x tasks)."""
    return stats.trim_mean(scores.reshape(*scores.shape[:-2], -1), 0.25, axis=-1)


def mean(scores: np.ndarray) -> np.ndarray:
    return scores.mean(axis=(-2, -1))


def stratified_bootstrap(matrix, statistic, reps: int, rng: np.random.Generator,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi): `statistic` of a runs x tasks matrix and its percentile CI. Each
    replicate resamples, with replacement, the runs of every task separately, never across
    tasks; `statistic` maps [..., runs, tasks] to [...]."""
    matrix = np.asarray(matrix, dtype=float)
    n_runs, n_tasks = matrix.shape
    replicates = matrix[rng.integers(0, n_runs, size=(reps, n_runs, n_tasks)), np.arange(n_tasks)]
    lo, hi = np.quantile(statistic(replicates), [alpha / 2, 1 - alpha / 2])
    return float(statistic(matrix)), float(lo), float(hi)


def _rng(seed: int, key) -> np.random.Generator:
    """One stream per table row, so a row's CI does not move when other rows come and go."""
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


# ─── Tables ─────────────────────────────────────────────────────────────────

def aggregate(tasks: pd.DataFrame, reps: int = REPS, seed: int = 0) -> pd.DataFrame:
    """IQM (and mean) over runs x tasks per (arch, finetune, checkpoint, block), a task being a
    member and a run one replicate (a source run's policy; the one rule-based policy)."""
    for name in BLOCK_NAMES:
        if not (tasks["block"] == name).any():
            notice(f"block {name}: no task scores, skipped")
    keys = ["arch", "finetune", "checkpoint", "block"]
    rows = []
    for key, group in tasks[tasks["block"].notna()].groupby(keys):
        per_member = group.groupby("member")
        runs = per_member.size()
        if runs.nunique() != 1:
            notice(f"aggregate {dict(zip(keys, key))} skipped: unequal runs per task "
                   f"{runs.to_dict()}")
            continue
        matrix = np.column_stack([g.sort_values("seed")["score"].to_numpy()
                                  for _, g in per_member])
        point, lo, hi = stratified_bootstrap(matrix, iqm, reps, _rng(seed, key))
        rows.append({**dict(zip(keys, key)), "iqm": point, "iqm_lo": lo, "iqm_hi": hi,
                     "mean": float(matrix.mean()), "runs": matrix.shape[0],
                     "tasks": matrix.shape[1], "members": ",".join(runs.index)})
    return pd.DataFrame(rows, columns=keys + ["iqm", "iqm_lo", "iqm_hi", "mean", "runs", "tasks",
                                              "members"])


def contrasts(tasks: pd.DataFrame, reps: int = REPS, seed: int = 0) -> pd.DataFrame:
    """Each CONTRASTS entry per (arch, checkpoint) of the policies trained on S (no fine-tuning):
    per source run, mean ZS over the targets - mean ZS over the baselines, both from that run's
    policy under that checkpoint rule; then the mean over runs and a bootstrap CI over runs."""
    zero_shot = tasks[(tasks["source"] == SOURCE) & ~tasks["checkpoint"].str.contains(">")]
    rows, skipped = [], {name: [] for name in CONTRASTS}
    for (arch, checkpoint), group in zero_shot.groupby(["arch", "checkpoint"]):
        per_run = group.pivot(index="source_run", columns="member", values="score")
        for name, (targets, baselines) in CONTRASTS.items():
            paired = per_run.reindex(columns=targets + baselines).dropna()
            if paired.empty:
                skipped[name].append(f"{arch}/{checkpoint}")
                continue
            if len(paired) < len(per_run):
                lacking = sorted(set(per_run.index) - set(paired.index))
                notice(f"contrast {name} ({arch}, {checkpoint}): runs {lacking} lack an "
                       f"evaluation on {targets + baselines}, left out")
            delta = (paired[targets].mean(axis=1) - paired[baselines].mean(axis=1)).to_numpy()
            point, lo, hi = stratified_bootstrap(delta[:, None], mean, reps,
                                                 _rng(seed, (arch, checkpoint, name)))
            rows.append({"arch": arch, "checkpoint": checkpoint, "contrast": name,
                         "targets": ",".join(targets), "baselines": ",".join(baselines),
                         "mean": point, "lo": lo, "hi": hi, "runs": len(delta)})
    for name, where in skipped.items():
        if where:
            notice(f"contrast {name} skipped for {', '.join(where)}: no run evaluated on all of "
                   f"{sum(CONTRASTS[name], [])}")
    return pd.DataFrame(rows, columns=["arch", "checkpoint", "contrast", "targets", "baselines",
                                       "mean", "lo", "hi", "runs"])


def few_shot(tasks: pd.DataFrame) -> pd.DataFrame:
    """One FS_k curve over FEW_SHOT_BUDGETS per run fine-tuned from S on the member (finetune
    full/head/extractor) or trained from scratch on it (finetune "scratch"). FS_k is the score of
    the run's model_at_<k> checkpoint, steps_<k> the env steps it was trained on. A fine-tuned
    run without its own model_at_0 takes FS_0 from the zero-shot score of the checkpoint it
    started from (the same parameters). AULC, the trapezoid over k / the largest k, is left
    empty for a curve that misses a budget."""
    columns = ["arch", "member", "finetune", "checkpoint", "seed", "run"] + \
        [f"fs_{b}" for b in FEW_SHOT_BUDGETS] + [f"steps_{b}" for b in FEW_SHOT_BUDGETS] + ["aulc"]
    k = tasks["checkpoint"].str.extract(r"(?:^|>)model_at_(\d+)$", expand=False).astype(float)
    curve_block = tasks["block"].isin(["few_shot", "scratch"])
    off_budget = k.notna() & ~k.isin(FEW_SHOT_BUDGETS) & curve_block
    if off_budget.any():
        notice(f"few-shot: model_at_<k> evaluations at k = "
               f"{sorted(k[off_budget].astype(int).unique())} are not budgets {FEW_SHOT_BUDGETS}:"
               f" in task_scores and aggregate only")
    at_budget = k.isin(FEW_SHOT_BUDGETS) & curve_block
    if not at_budget.any():
        return pd.DataFrame(columns=columns)
    rows = tasks[at_budget].assign(
        k=k[at_budget].astype(int),
        start=tasks.loc[at_budget, "checkpoint"].str.rpartition(">")[0],   # "" from scratch
        finetune=tasks.loc[at_budget, "finetune"].where(tasks.loc[at_budget, "block"] == "few_shot",
                                                         "scratch"))
    keys = ["arch", "member", "finetune", "start", "seed", "run", "source_run"]
    wide = rows.set_index(keys + ["k"])[["score", "trained_steps"]].unstack("k")
    fs = wide["score"].reindex(columns=list(FEW_SHOT_BUDGETS)).astype(float)
    steps = wide["trained_steps"].reindex(columns=list(FEW_SHOT_BUDGETS)).astype(float)
    started = tasks.set_index(["run", "checkpoint", "member"])["score"]
    for i, (_, member, finetune, start, _, _, source_run) in enumerate(fs.index):
        origin = (source_run, start, member)
        if finetune != "scratch" and np.isnan(fs.iat[i, 0]) and origin in started:
            fs.iat[i, 0], steps.iat[i, 0] = started[origin], 0
    complete = fs.notna().all(axis=1).to_numpy()
    if not complete.all():
        notice(f"few-shot: {int((~complete).sum())} curves miss one of the budgets "
               f"{FEW_SHOT_BUDGETS}, AULC left empty: "
               + ", ".join(sorted(fs.index[~complete].get_level_values("run"))))
    aulc = integrate.trapezoid(fs.to_numpy(), x=FEW_SHOT_BUDGETS, axis=1) / FEW_SHOT_BUDGETS[-1]
    out = pd.concat([fs.set_axis([f"fs_{b}" for b in FEW_SHOT_BUDGETS], axis=1),
                     steps.set_axis([f"steps_{b}" for b in FEW_SHOT_BUDGETS], axis=1)
                     .astype("Int64")], axis=1)
    out["aulc"] = np.where(complete, aulc, np.nan)
    out = out.reset_index()
    out["checkpoint"] = np.where(out["start"] == "", "model_at_k", out["start"] + ">model_at_k")
    return out[columns]


def transfer_vs_scratch(curves: pd.DataFrame) -> pd.DataFrame:
    """Fine-tuned runs against the runs of the same arch trained from scratch on the member,
    over complete curves: per (arch, member, finetune, checkpoint) the FS_k gain (mean fine-tuned
    - mean scratch) at each budget, both mean AULCs and the area ratio (A_transfer - A_scratch) /
    A_scratch, left empty unless A_scratch > 0 (a ratio to an area at or below the random floor
    means nothing)."""
    complete = curves[curves["aulc"].notna()]
    scratch = complete[complete["finetune"] == "scratch"]
    rows = []
    for (arch, member, finetune, checkpoint), tuned in complete[complete["finetune"] != "scratch"] \
            .groupby(["arch", "member", "finetune", "checkpoint"]):
        base = scratch[(scratch["arch"] == arch) & (scratch["member"] == member)]
        if base.empty:
            notice(f"transfer vs scratch: no complete scratch curve of {arch} on {member}, "
                   f"skipped")
            continue
        a_tuned, a_scratch = tuned["aulc"].mean(), base["aulc"].mean()
        if not a_scratch > 0:
            notice(f"transfer vs scratch ({arch}, {member}): scratch AULC {a_scratch:.4f} <= 0, "
                   f"area ratio left empty")
        rows.append({"arch": arch, "member": member, "finetune": finetune,
                     "checkpoint": f"{checkpoint} vs model_at_k",
                     **{f"gain_{b}": tuned[f"fs_{b}"].mean() - base[f"fs_{b}"].mean()
                        for b in FEW_SHOT_BUDGETS},
                     "aulc_transfer": a_tuned, "aulc_scratch": a_scratch,
                     "area_ratio": (a_tuned - a_scratch) / a_scratch if a_scratch > 0 else np.nan,
                     "runs_transfer": len(tuned), "runs_scratch": len(base)})
    return pd.DataFrame(rows, columns=["arch", "member", "finetune", "checkpoint"]
                        + [f"gain_{b}" for b in FEW_SHOT_BUDGETS]
                        + ["aulc_transfer", "aulc_scratch", "area_ratio", "runs_transfer",
                           "runs_scratch"])


# ─── Output ─────────────────────────────────────────────────────────────────

def check_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Every reported table names, in every row, the checkpoint rule its numbers come from."""
    for name, table in tables.items():
        if "checkpoint" not in table.columns:
            raise ValueError(f"table {name} has no checkpoint column")
        unnamed = table["checkpoint"].isna() | (table["checkpoint"].astype(str).str.strip() == "")
        if unnamed.any():
            raise ValueError(f"table {name}: {int(unnamed.sum())} rows name no checkpoint rule")


def write_tables(tables: dict[str, pd.DataFrame], out: str) -> None:
    """One CSV per table, and none unless every table passes check_tables. A TABLES name
    without a table here has nothing to report: its CSV from an earlier call is removed, so
    no stale table sits next to fresh ones."""
    check_tables(tables)
    os.makedirs(out, exist_ok=True)
    for name, table in tables.items():
        path = os.path.join(out, f"{name}.csv")
        table.to_csv(path, index=False)
        print(f"wrote {path} ({len(table)} rows)")
    for name in TABLES:
        path = os.path.join(out, f"{name}.csv")
        if name not in tables and os.path.exists(path):
            os.remove(path)
            notice(f"removed {path}, written by an earlier call: nothing to report now")


def _keep(tables: dict, name: str, table: pd.DataFrame, why: str) -> None:
    if table.empty:
        notice(f"{name}: {why}, not written")
    else:
        tables[name] = table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="The paper's tables from evaluation.csv files.")
    parser.add_argument("--logs", required=True, help="base_log_dir holding the evaluate runs")
    parser.add_argument("--references", required=True, help="references_summary_<split>.csv")
    parser.add_argument("--out", required=True, help="directory for the CSV tables")
    parser.add_argument("--reps", type=int, default=REPS, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args(argv)

    refs, split = load_references(args.references)
    scored = score_levels(load_evaluations(args.logs), refs, split)
    if scored.empty:
        raise ValueError(f"no evaluation under {args.logs} can be scored against "
                         f"{args.references} (see the notices above); nothing written")
    tasks = task_scores(scored, split)
    outside = tasks["block"].isna()
    if outside.any():
        notice(f"{int(outside.sum())} task scores are in no block (transfer chains, other "
               f"sources, off-target evaluations): task_scores.csv only")
    trained_on_source = tasks[(tasks["source"] == SOURCE) & ~tasks["checkpoint"].str.contains(">")]
    for arch, group in trained_on_source.groupby("arch"):
        absent = sorted({"best_val_model", "final_model"} - set(group["checkpoint"]))
        if absent:
            notice(f"{arch}: no {absent} evaluations; the tables should report both rules")

    tables = {"task_scores": tasks}
    _keep(tables, "aggregate", aggregate(tasks, args.reps, args.seed),
          "no block could be aggregated (see the notices above)")
    _keep(tables, "contrasts", contrasts(tasks, args.reps, args.seed),
          "no policy trained on S is evaluated on every member of a contrast")
    curves = few_shot(tasks)
    _keep(tables, "few_shot", curves, "no model_at_<k> evaluations of fine-tuned or scratch runs")
    _keep(tables, "transfer_vs_scratch", transfer_vs_scratch(curves),
          "no complete fine-tuned and scratch curves on a common member")
    write_tables(tables, args.out)
    for name in ("aggregate", "contrasts"):
        if name in tables:
            print(f"\n{name}:\n" + tables[name].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()

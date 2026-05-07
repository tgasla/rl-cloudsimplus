# Extractor Tournament — Experiment Plan

## Goal

Compare three feature extractors for the PPO-X job-placement agent:

| Extractor | Architecture | Key claim |
|-----------|-------------|-----------|
| `euromlsys` | Categorical embedding + dual MLP + residual adaptation | Conference baseline (PPO-X) |
| `turret` | Graph Attention Network (GATConv) over DC topology | Published TURRET baseline |
| `attention` | Transformer: host self-attention → DC scatter-mean → cross-attention with jobs → global encoder | New contender |

Each extractor is trained on **Env B** (3 cloud DCs, 5 total DCs), then transferred to:
- **Env A** — downscale (2 DCs, no cloud tier) — easier transfer
- **Env C** — upscale (5 DCs, 2 new micro-DCs) — harder transfer

Oracle experiments (scratch training in the target env) provide the normalization ceiling
needed to evaluate transfer quality independently of training quality.

---

## Experiment Matrix

### Phase 1 — Train on Env B (source domain)

| # | Name | Status | Best reward |
|---|------|--------|-------------|
| — | `euromlsys_b_train` | ✅ done | 12.64 |
| 1 | `turret_b_train` | 🔄 **re-running** (partial: 104k/400k steps) | 10.13 (partial) |
| — | `attention_b_train` | ✅ done | 12.68 |

> **Note**: `turret_b_train` was killed at ~104k steps (container stop). Re-running to completion
> before any turret transfer experiments can be trusted.

### Phase 2 — Transfer B → A (downscale, euromlsys + attention)

| # | Name | Source model | Extractor |
|---|------|-------------|-----------|
| 2 | `euromlsys_b_to_a` | `euromlsys_b_train` | euromlsys |
| 3 | `attention_b_to_a` | `attention_b_train` | attention |

### Phase 3 — Transfer B → C (upscale, euromlsys + attention)

| # | Name | Source model | Extractor |
|---|------|-------------|-----------|
| 4 | `euromlsys_b_to_c` | `euromlsys_b_train` | euromlsys |
| 5 | `attention_b_to_c` | `attention_b_train` | attention |

### Phase 4 — Oracle baselines (non-turret)

Train from scratch in the target environment. These define the performance ceiling
for normalizing transfer metrics.

| # | Name | Env | Extractor |
|---|------|-----|-----------|
| 6 | `oracle_a_euromlsys` | A | euromlsys |
| 7 | `oracle_a_attention` | A | attention |
| 8 | `oracle_c_euromlsys` | C | euromlsys |
| 9 | `oracle_c_attention` | C | attention |

### Phase 5 — Turret transfers + oracles (after turret re-train completes)

| # | Name | Source model / Env | Extractor |
|---|------|-------------------|-----------|
| 10 | `turret_b_to_a` | `turret_b_train` | turret |
| 11 | `turret_b_to_c` | `turret_b_train` | turret |
| 12 | `oracle_a_turret` | A | turret |
| 13 | `oracle_c_turret` | C | turret |

### Run command

All 13 experiments run sequentially with a single command:

```bash
make run domain=job-placement
```

Check progress any time:

```bash
python3 compare_results.py
```

---

## Reward Logging — Which Source for What

Each experiment writes three reward sources to its log directory:

### 1. `monitor.csv` — raw per-episode rewards

Written by SB3's `Monitor` wrapper. One row per completed episode, from **all** parallel workers.

```
r,l,t
7.24,24,2.50   ← episode reward, length, wall time
6.35,24,2.50
...
```

**Use for:**
- Zero-shot jumpstart (first episode in the transfer env)
- Per-episode AUC (high resolution, no smoothing bias)
- Time-to-threshold (episode-level accuracy)
- Final convergence: mean of last N episodes

### 2. `progress.csv` — mixed per-episode and per-rollout rows

Written by `SaveOnBestTrainingRewardCallback`. Two interleaved row types:

- **Per-episode rows** (`train/*` columns): `train/ep_total_rew`, `train/ep_jobs_placed_ratio`,
  `train/ep_quality_ratio`, `train/ep_deadline_violation_ratio`, etc. Only worker 0 is recorded.
- **Per-rollout rows** (`rollout/*` columns): `rollout/ep_rew_mean` = rolling mean over the last
  100 completed episodes from all workers (computed by SB3 from Monitor). Also includes per-metric
  rolling means.

**Use for:**
- Smooth training curves (rollout/ep_rew_mean)
- Per-component reward breakdown (jobs placed %, quality %, deadline violation %)

### 3. `events.out.tfevents.*` — TensorBoard

Contains the `rollout/*` metrics (same as progress.csv rollout rows). Does NOT contain
per-episode `train/*` rows (those are excluded with `exclude="tensorboard"`).

```bash
make run-tensorboard    # start TensorBoard at http://localhost:6006
```

### Why not use `rollout/ep_rew_mean` for transfer metrics?

`rollout/ep_rew_mean` is a rolling 100-episode window. At the start of a transfer experiment:
- The first rollout completes ~80 episodes (128 steps/rollout × 16 workers ÷ ~24 steps/episode)
- `rollout/ep_rew_mean` at rollout 1 = mean of those ~80 episodes, not episode 1

This means the "first rollout reward" already averages away the true zero-shot signal.
Use `monitor.csv` row 0 for the real jumpstart.

---

## Transfer Evaluation Metrics

For each (extractor, direction) pair, we compute three normalized metrics using `compare_results.py`.
All normalization is relative to the oracle (scratch training in the same target env).

### 1. Normalized Jumpstart

```
norm_jumpstart = first_episode_reward_in_transfer / oracle_peak_reward
```

- Source: `monitor.csv` row 0 for transfer; `max(monitor.csv r)` for oracle
- Interpretation: how close to ceiling the agent lands **before any fine-tuning**
- Range: 0 (random) to ~1 (perfect zero-shot transfer)

### 2. AUC Ratio

```
auc_ratio = AUC(transfer monitor curve) / AUC(oracle monitor curve)
```

Both curves are raw per-episode rewards. AUC is trapezoidal, normalized by episode count.
Only the first N episodes are used (N = min of transfer and oracle episode counts).

- Interpretation: does the warm-start accumulate more reward per episode than training from scratch?
- Values > 1 = positive transfer; < 1 = neutral or negative transfer

### 3. Time-to-Threshold

```
steps_to_80pct = first episode index where reward >= 0.8 × oracle_peak
```

- Source: `monitor.csv`
- Interpretation: how quickly the agent recovers to near-oracle performance
- Practical relevance: in real deployments, "time to acceptable performance" matters more than
  asymptotic reward

### Summary Table (printed by compare_results.py)

```
Extractor    │ Direction │ Jumpstart │ NormJump │ AUC-Ratio │ Ep-to-80%
─────────────┼───────────┼───────────┼──────────┼───────────┼──────────
euromlsys    │ B→A       │ 9.2       │ 0.74     │ 1.12      │ 18
turret       │ B→A       │ 8.1       │ 0.65     │ 0.98      │ 31
attention    │ B→A       │ 10.3      │ 0.83     │ 1.28      │ 11
```

Note: oracle experiments must complete before NormJump, AUC-Ratio, and Ep-to-80% are populated.
Until then, compare_results.py shows `—(no oracle)`.

---

## Expected Results (Hypotheses)

### Training (Env B)

- `attention ≈ euromlsys > turret` — transformer cross-attention and GNN both exploit topology
  structure, but turret's graph construction overhead slows training wall-time.

### Transfer B→A (downscale — easier)

- All three should transfer well; performance close to oracle.
- `attention` expected to have higher norm-jumpstart because the DC scatter-mean aggregation is
  invariant to DC count (fewer real DC slots, same padding behavior).

### Transfer B→C (upscale — harder)

- `turret`: GNN explicitly encodes DC-to-DC edges; adding new DCs changes the graph structure,
  causing representation shift. Expect lower jumpstart but decent recovery.
- `euromlsys`: residual adaptation layer helps but the base embedding shifts with new DC IDs.
- `attention`: padding-invariant scatter-mean + learned cross-attention should generalize best.
  Zero-padded slots are masked during host self-attention, so adding real DC slots changes the
  aggregation distribution less than it changes mean-pooling.

---

## Seed & Reproducibility

Current runs use `seed: 1234` (set in `config.yml → common → seed`).
For the paper, run each variant with at least 3 seeds (1234, 2345, 3456) and report mean ± std.
Update the `seed` key in each experiment block to vary seeds.

---

## How to Read `compare_results.py`

```bash
# From repo root:
python3 compare_results.py

# For a different log root:
python3 compare_results.py common/logs/euromlsys/extractor_comparison
```

Output sections:
1. **Training leaderboard** — rollout/ep_rew_mean best and final for all experiments
2. **Transfer metrics table** — jumpstart, AUC ratio, steps-to-80% (requires oracle runs)
3. **Other experiment groups** — any other dirs in `common/logs/euromlsys/`

---

## Directory Structure

```
common/logs/euromlsys/extractor_comparison/
├── euromlsys_b_train/       ← training run (source)
│   ├── monitor.csv          ← raw episode rewards (all workers)
│   ├── progress.csv         ← per-episode + per-rollout stats
│   ├── events.out.tfevents* ← TensorBoard
│   └── best_model.zip       ← checkpoint used for transfer
├── turret_b_train/          ← (re-running)
├── attention_b_train/
├── euromlsys_b_to_a/        ← transfer run (target env A)
├── turret_b_to_a/
├── attention_b_to_a/
├── euromlsys_b_to_c/        ← transfer run (target env C)
├── turret_b_to_c/
├── attention_b_to_c/
├── oracle_a_euromlsys/      ← scratch training in env A (normalization denominator)
├── oracle_a_turret/
├── oracle_a_attention/
├── oracle_c_euromlsys/      ← scratch training in env C
├── oracle_c_turret/
└── oracle_c_attention/
```

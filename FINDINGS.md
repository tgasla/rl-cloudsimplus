# Key Findings & Observations

Engineering and empirical discoveries from running the job-placement RL training system.
Ordered by theme. Dates are when the finding was confirmed.

---

## Training Speed & FPS

### GPU vs CPU (2026-05-07)

Tested `gpu: true` vs `gpu: false` on the job-placement MaskablePPO pipeline (16 workers, RTX 4080).

| Phase | CPU FPS | GPU FPS | Winner |
|---|---|---|---|
| Rollout 1 (no policy update yet) | 241 | 214 | CPU (+13%) |
| Rollout 2+ (policy update active) | 39 | 132 | **GPU (3.4×)** |

**Interpretation**: The first rollout is pure simulation + data collection — no gradient computation.
CPU wins here because there is no CUDA kernel launch overhead. As soon as the PPO policy update
starts (Adam optimizer + multiple epochs over minibatches), GPU wins decisively. Despite the
architecture having no convolutional layers (MLP + embeddings only), the optimizer step is
compute-heavy enough that GPU dominates at steady state.

**Conclusion**: Always use `gpu: true` for this system.

---

### FPS Decay Curve Shape

The `time/fps` metric in SB3 is a **cumulative average** (`total_timesteps / total_elapsed_time`),
not an instantaneous measurement. This produces a characteristic smooth decay curve even when
the real throughput is roughly constant at steady state.

The specific decay shape observed here has two compounding causes:

1. **Training dynamics**: Early in training, a random policy places all 50 trace jobs in ~10–20
   steps per episode (queue empties quickly → `isRunning()` returns false → reset). As the agent
   learns, it becomes more selective, leading to longer episodes. More time per episode = lower
   cumulative average FPS.

2. **Java reset cost per rollout**: Each episode reset rebuilds the entire CloudSim simulation
   from scratch (new `CloudSimPlus`, broker, datacenters, hosts, VMs). As episodes shorten over
   training, the number of resets per rollout (`n_steps / episode_length`) increases — but here
   episodes get *longer* with training (agent deliberates more), so this compounds effect 1.

The FPS curve asymptotes when episode length stabilizes at the learned policy's typical length.
This is expected training dynamics, not a bug.

**Metric to watch in TensorBoard**: `train/episode_length` — if it increases over training,
that confirms the explanation above.

---

### Eliminated Bottlenecks (cumulative)

| Bottleneck | Root cause | Fix | Impact |
|---|---|---|---|
| `load_results()` CSV parsing | Read+parse entire Monitor CSV from disk every episode end. O(episodes²) total work | Replaced with `np.sum(self.rewards)` — O(1), in-memory | High — primary cause of smooth FPS decay in early experiments |
| `UseSerialGC` in JVM | Stop-the-world GC pauses growing as heap fills across episodes | Replaced with `-XX:+UseG1GC -Xmx256m` | Medium — bounded GC pause time |
| Per-step GPU→CPU tensor copies in callback | Dead `self.observations`/`self.new_observations` fields collected every step but never written anywhere | Removed fields and `_get_observation_from_locals()` | Medium — 2 tensor copies per step eliminated |
| `action_masks()` Python loops | O(n²) pure Python loop over hosts × jobs | Vectorized with `np.maximum.at` scatter-max + numpy broadcasting — O(n) | Medium — called every step by MaskablePPO |

---

### TURRET Throughput

TURRET's GNN forward pass (GATConv graph attention over DC topology) averages **~20 FPS**
before optimization — approximately 5× slower than PPO-X (`euromlsys`, ~100 FPS) and the
`attention` extractor (~100 FPS).

**Root cause — three compounding bottlenecks:**

1. **Python `for b in range(batch_size):` loop** (dominant). The original `forward()` iterated
   over every sample in the minibatch individually in Python. Each iteration calls GATConv
   from Python, paying interpreter overhead × batch_size. With batch_size=64, this means
   64 sequential Python-dispatched GATConv invocations instead of one vectorized GPU call.

2. **Dynamic `edge_index` construction every call**. Inside the loop, a fully-connected
   `edge_index` was rebuilt via `torch.meshgrid + mask + torch.stack` on every forward pass,
   for every sample. For n = max_hosts + max_jobs ≈ 80 nodes, that is 6,320 edges
   re-allocated ~64 × (many calls/second) times. The graph topology never changes.

3. **GATConv attention complexity**. Each GATConv layer runs O(E × heads) attention where
   E = n×(n−1) = 6,320 for n=80. With concat=True and 4 heads × 2 layers, the full
   attention kernel runs 8 times per sample per forward call — all serialized in Python.

**Fix** (implemented 2026-05-07):
- `edge_index` precomputed once in `__init__` and registered as a PyTorch buffer
  (moves to GPU automatically; never re-allocated).
- Python loop replaced: all batch samples are merged into one large graph by offsetting
  each sample's node indices by `b × n`, producing a single `[2, B×E]` edge tensor.
  GATConv is called once on `[B×n, D]` node features — pure GPU vectorization.
- `_embed_host_nodes()` helper inlined and batched: embedding lookups now operate on
  `[B, H]` index tensors, not `[H]` slices from a loop.

**Expected improvement**: 20 FPS → 80–100 FPS (GPU) after vectorization.

**Paper significance**: Even after optimization, TURRET's GNN forward pass is architecturally
heavier than PPO-X's MLP extractor (O(E × heads) vs. O(H × D) per sample). PPO-X is not only
competitive in transfer quality but also significantly more inference-efficient at steady state,
which matters for real-time deployment.

### TURRET Policy Update Bottleneck

After the vectorized rollout fix (123 FPS at rollout), the policy update phase became the
dominant cost. Root cause: the GATConv message tensor `[E, heads, out_ch]` ≈ 140 MB per sample
must be retained by PyTorch autograd for ALL minibatch samples simultaneously during backward.
With batch_size=64 and n_epochs=10: 64 × 140 MB × 10 epochs ≈ 9 GB storage + recomputation.

**Fixes applied:**
- `torch.utils.checkpoint.checkpoint()` wrapping `_gnn_forward()` during `self.training=True`.
  Discards GATConv intermediates after forward; recomputes them sample-by-sample during backward.
  Reduces peak memory from `B × 140 MB` to `~140 MB` at cost of ~2× backward compute.
- `n_epochs: 2` for TURRET experiments (wired `n_epochs` into `misc.py` algorithm_kwargs).
  Reduces policy update passes from 320 to 64 per rollout (~5× speedup on update phase).

**Estimated training time at 600k steps** (n_rollout_steps=128, n_envs=16, 293 rollouts):
| Config | Rollout/rollout | Update/rollout | Total |
|--------|-----------------|----------------|-------|
| TURRET, n_epochs=10 | 16s | ~5–10 min | ~25–50 hrs |
| TURRET, n_epochs=2 | 16s | ~60–120s | ~5–10 hrs |
| PPO-X, n_epochs=10 | 16s | ~2s | ~1.5 hrs |

**Comparison fairness**: Using different n_epochs for different extractors biases learning curves
(fewer gradient updates per step → slower convergence). For the paper's architecture comparison,
all extractors should use the same n_epochs. Use n_epochs=2 for all (accommodates TURRET), and
separately document PPO-X's full performance at n_epochs=10 to show the difference.

**Parameter tuning levers for TURRET speed (diminishing returns):**
- `n_epochs` — **biggest lever**: linear speedup on update phase. 10→2 is 5×.
- `num_cpu` — speeds rollout only (rollout is already cheap at 125 FPS); update is GPU-bound.
- `batch_size` — **no effect**: GATConv is per-sample sequential; total GATConv work = B × n per
  minibatch regardless of batch_size. Larger batch = proportionally longer minibatch = same total.
- `n_rollout_steps` — **no direct effect**: scaling up with batch_size proportionally keeps
  minibatch count constant but doubles rollout time (which is cheap). Marginal improvement
  to rollout/update ratio but doesn't reduce absolute update time.

### EuroMLSys FPS Ramp-Up (247 → 317 FPS over training)

Unlike TURRET (flat ~20 FPS), the `euromlsys` extractor's FPS **rises** during a training run.
The mechanism:

- Early training: random policy places all jobs in ~10–20 steps per episode (no discrimination
  — just dumps everything). Episodes are short → many resets per rollout → Java infrastructure
  rebuild cost dominates.
- Later training: the agent learns selective placement. Episode steps increase as it evaluates
  options. But once jobs ARE placed, the queue empties faster → fewer total simulation clock
  advances per episode. The Java side processes fewer events per placed cloudlet.
- The cumulative FPS average (`total_steps / elapsed`) rises as the per-step wall time drops.

This is policy learning changing the simulation dynamics, not the extractor getting faster.
TURRET shows no similar ramp-up because its bottleneck (Python loop overhead) is constant
regardless of episode length — it saturates the interpreter, not the simulation.

---

## Algorithmic Findings

### Episode Termination — Drain Phase Elimination

**Problem**: The original `isRunning()` in `CloudSimProxyBase` returned true until
`broker.getCloudletFinishedList().size() >= inputJobs.size()` — i.e., until all submitted
cloudlets *finished executing*. Since cloudlets can execute for many simulated minutes after
placement, each episode had a long "drain phase" where the agent was receiving zero reward but
the simulation kept running.

**Fix**: Overrode `isRunning()` in `CloudSimProxy` (job-placement) to terminate when
`jobQueue.isEmpty()` — all jobs have been placed (removed from the queue by the agent's action).

```java
@Override
public boolean isRunning() {
    return cloudSimPlus.isRunning() && !jobQueue.isEmpty();
}
```

**Effect**: Episodes now end when all jobs are placed, not when they finish executing.
Reward is computed entirely at placement time, so no signal is lost. Episode length dropped
significantly, and training throughput improved.

---

### Action Masking — Real Implementation

The original `action_masks()` was a stub returning all-True. It was replaced with
a correct vectorized implementation:

1. From `_last_infr_obs`, compute the max free PEs per DC using `np.maximum.at` scatter-max.
2. Broadcast `[max_jobs, 1]` requested cores against `[1, max_datacenters]` capacity.
3. Special cases: action=0 (no-op) always valid; padding job slots (cores=0) allow all;
   fully-blocked jobs fall back to all-valid (MaskablePPO invariant: ≥1 valid action per sub-space).

The mask is recomputed every step using `_last_infr_obs` and `_last_jobs_obs` cached in
`_get_observation()`.

---

### Two-Stage Placement Model

Job placement operates in two stages:

| Stage | Decision | Who decides |
|---|---|---|
| 1 — macro | Which **datacenter** runs this cloudlet? | RL agent (when `cloudlet_to_dc_mapping: rl`) |
| 2 — micro | Which **VM within that DC** runs it? | Fixed rule (`most-free-pes`) |

The RL agent operates **only on stage 1**. Stage 2 is always rule-based. CloudSim Plus has no
native cloudlet→DC concept; this two-stage model is a higher-level abstraction layered on top.

---

### Reproducibility Settings

`set_seed_for_all()` in `entrypoint.py` sets seeds for `random`, `numpy`, `torch`, CUDA, and
`PYTHONHASHSEED`. The key findings:

- `cudnn.deterministic=True` and `benchmark=False` are **no-ops** for this architecture
  (no Conv layers) — kept for safety but have zero effect.
- `use_deterministic_algorithms(True, warn_only=True)` was removed because with `warn_only=True`
  Flash Attention still ran non-deterministically (the warning was produced with no benefit).
- Replaced with `enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)` — forces
  math-only attention, which is deterministic and eliminates the warning.
- `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` are correct for multi-process RL to prevent
  OpenMP thread contention across 16 workers.

None of these settings cause time-varying FPS overhead — all overhead is constant from startup.

---

## Infrastructure Findings

### Docker Profile Isolation

`docker-compose.yml` services without a `profiles:` key are always started (the "default" group).
When `gpu: true` was configured, both `manager` (CPU) and `manager-cuda` (GPU) started
simultaneously, causing a GPU crash.

**Fix**: Added `profiles: ["cpu"]` to the `manager` service so it is only started explicitly
via `--profile cpu`. The `make stop` target uses `--profile cpu --profile cuda` to stop either.

---

### Java Reset Cost (Pending Investigation)

Each episode reset calls `WrappedSimulationBase.reset()` which creates a brand-new CloudSim
simulation from scratch:
- New `CloudSimPlus` instance
- New `DatacenterBrokerFirstFitFixed`
- New datacenters, hosts, VMs (from topology config)
- `cloudSimPlus.startSync()` + initial `proceedClockTo(minTimeBetweenEvents)`

This cost is fixed per reset but amortized over episode length. As the agent learns and episode
lengths increase (more deliberate placements), more time is spent in the episode relative to
setup. However, if episode lengths *decrease* (e.g., after a design change), reset overhead
could dominate throughput.

**Future direction**: Replace full reconstruction with a lightweight reset — requeue cloudlets,
clear broker finished/submitted lists, reset clock — without recreating CloudSim objects.

---

### G1GC for JVM Subprocesses

Each of the 16 simulation workers spawns a JVM subprocess. The original JVM flag was
`-XX:+UseSerialGC`, which performs stop-the-world full-heap collections. As each episode
creates and GC's hundreds of CloudSim objects (DCs, VMs, cloudlets), heap fills and
SerialGC pauses grew proportionally — causing smooth FPS decay within the session.

Replaced with `-XX:+UseG1GC -Xmx256m`:
- G1GC does concurrent incremental collection with bounded pause times
- 256MB heap cap prevents the process from accumulating stale objects indefinitely

---

## Running Experiments

### Detached SSH-safe training

```bash
# Set attached: false in config.yml first
nohup make run domain=job-placement > ~/training_run.log 2>&1 &
echo "PID: $!"

# Monitor orchestration progress
tail -f ~/training_run.log

# Monitor live Python output
docker compose -f common/docker-compose.yml --profile cuda logs -f manager-cuda

# Monitor metrics
make run-tensorboard domain=job-placement
```

Docker containers are managed by the Docker daemon independently — they keep running even if
the shell script is killed. `nohup` protects the `run_docker.sh` orchestration loop
(multi-experiment sequencing) from SIGHUP if the SSH session drops.

### Terminal output does not affect training speed

Docker captures container stdout to its json-file log driver regardless of whether anyone
is following the logs. `attached: true` vs `false` only affects whether the host terminal
renders those logs — the container's I/O is identical in both cases.

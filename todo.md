# TODO

## Completed

- [x] **Fix `send_observation_tree_array` flag propagation** — flag now flows config.yml → train.py/transfer.py → callback → all write sites.
- [x] **Guard tree array computation in Java** — when flag=false, SimulationStepInfo skips `getInfrastructureObservation()`.
- [x] **Use `shadowJar` for fat JAR** — `make build-gateway` uses `shadowJar` correctly.
- [x] **Remove Py4J** — all Py4J gym env and orchestration code removed.
- [x] **Remove batch/parallel mode** — `run_mode: serial` only.
- [x] **Lombok migration** — `@Value`/`@Getter`/`@Setter` on all value objects.
- [x] **All System.out/err replaced with SLF4J**.
- [x] **Remove `MultiSimulationEnvironment.java`**.
- [x] **Centralized version management** — `versions.gradle` is single source of truth.
- [x] **Docker reproducibility** — `.dockerignore`, deterministic pip installs.
- [x] **Per-experiment Java logs** — runtime `logback.xml` per experiment ID.
- [x] **Implement real action masking (MaskablePPO)** — vectorized with `np.maximum.at` scatter-max + numpy broadcasting. O(n) not O(n²). `rl_algorithm: MaskablePPO` in config.
- [x] **Replace UseSerialGC with G1GC** — `UseG1GC -Xmx256m` in both Java spawn paths in `misc.py`. SerialGC caused stop-the-world pauses that grew as heap filled across episodes.
- [x] **Fix `load_results()` O(episodes²) bottleneck** — callback now uses `np.sum(self.rewards)` (O(1), in-memory) instead of reading and parsing the full Monitor CSV from disk every episode end.
- [x] **Remove per-step GPU→CPU tensor copies from callback** — dead `self.observations`/`self.new_observations` accumulation removed; `_get_observation_from_locals()` deleted.
- [x] **GPU profile isolation** — `profiles: ["cpu"]` on manager service; explicit `--profile` flags throughout `run_docker.sh` and `Makefile`. Both services no longer start simultaneously.
- [x] **`isRunning()` override for episode termination** — job-placement `CloudSimProxy.isRunning()` terminates on `jobQueue.isEmpty()` (all jobs placed), not when all jobs finish executing. Eliminates drain phase.
- [x] **Java 25 download URL fix** — Dockerfile tag corrected from `jdk-25.0.2%2B10` to `jdk-25.0.3%2B9`.
- [x] **Flash Attention determinism** — replaced `use_deterministic_algorithms(True, warn_only=True)` with `enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)`. Eliminates warning, enforces deterministic math-only attention backend.

## Active Experiments

- [x] **CPU vs GPU training speed** — tested 2026-05-07. **GPU wins decisively.**
  - Rollout 1 (pure collection, no policy update yet): CPU 241 FPS vs GPU 214 FPS — CPU slightly faster (no CUDA overhead)
  - Rollout 2+ (policy update active): CPU drops to **39 FPS**, GPU holds at **132 FPS**
  - Despite no Conv layers, Adam optimizer + multiple PPO epochs over minibatches is heavy enough that GPU wins 3.4× after warmup. Use `gpu: true`.

## Experiment Queue

- [ ] **Run all per-extractor oracle experiments** — each extractor must be normalized against
  its own oracle (self-normalization principle). Using a shared oracle biases normalized metrics:
  if `attention` achieves a higher ceiling from scratch, its norm-jumpstart is understated.
  Required runs (scratch training in target env, same step budget as transfer):
  - `oracle_a_attention` — attention from scratch in Env A
  - `oracle_a_turret` — turret from scratch in Env A
  - `oracle_c_euromlsys` — euromlsys from scratch in Env C
  - `oracle_c_attention` — attention from scratch in Env C
  - `oracle_c_turret` — turret from scratch in Env C
  All are in Phase 4/5 of the config matrix. Once done, update `transfer_analysis.py`
  to use per-extractor oracle as denominator per method.

## Performance Investigations

- [ ] **Java lightweight reset** — `WrappedSimulationBase.reset()` currently rebuilds the entire CloudSim infrastructure from scratch (new `CloudSimPlus`, broker, datacenters, hosts, VMs). As the agent learns and episodes shorten (fewer steps per episode → more resets per rollout), reset overhead dominates. Fix: instead of creating new objects, reset state of existing objects (requeue cloudlets, clear broker lists, reset clock). Significant Java refactor but would eliminate the reset cost growing with training progress.

- [x] **TURRET FPS root cause + fix** — Three compounding bottlenecks identified: (1) Python `for b in range(batch_size)` loop strips GPU vectorization; (2) fully-connected `edge_index` rebuilt from scratch every call via `torch.meshgrid`; (3) GATConv called B times from Python instead of once. Fix: precompute `edge_index` as a registered buffer, offset node indices per sample, flatten batch into single `[B*n, D]` graph call. Expected: 20 FPS → 80–100 FPS. Still inherently heavier than PPO-X MLP — paper comparison point remains valid.

## Pending Optimizations

- [ ] **Guard `getInfrastructureObservation()` double-call** — `step()` calls it twice per step (once for `SimulationStepInfo`, once for `Observation`). Compute once and pass to both. Low priority.
- [ ] **Batch gRPC calls** — send N steps per roundtrip; reduces roundtrip count. Requires proto schema changes.
- [ ] **Optional proto field** — make `observation_tree_array` optional in proto so Java skips sending when flag=false. Requires proto recompile both sides.

## Architecture Notes (current state)

- **16 JVMs**: each spawned as subprocess by `spawn_java_gateway()` in `misc.py`, listening on ports 50051–50066. G1GC with 256MB heap cap.
- **SubprocVecEnv**: 16 parallel workers. Each worker has its own JVM subprocess and gRPC channel.
- **Episode termination**: `isRunning() = cloudSimPlus.isRunning() && !jobQueue.isEmpty()`. Episode truncated at `max_episode_length: 150` steps if queue not yet empty.
- **Remaining FPS decay explanation**: the `time/fps` metric in SB3 is cumulative average (`total_steps / elapsed_time`). Early in training, random policies place all jobs in ~10-20 steps (short episodes). Trained policies make more selective placements (longer episodes, more resets per rollout). FPS curve asymptotes when episode length stabilizes. This is expected training dynamics, not a bug.
- **Action masking**: vectorized scatter-max over `_last_infr_obs` and `_last_jobs_obs`. Called every step by MaskablePPO.
- **Callback**: per-step tracking uses Python lists (cleared each episode). Metric deques are `maxlen=100`. No unbounded accumulation.

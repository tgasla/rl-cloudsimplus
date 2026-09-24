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

## Experiment Queue — Phase Next (post Key Finding 6)

### Priority 1 — Fusion high-budget run (decisive paper experiment)

- [ ] **Train fusion on Env B at 300k steps** (3× current) with same hyperparams. Then transfer
  to Env A and Env C at 50k each (same budget as all prior experiments — keeps oracle comparison fair).
  Expected: source policy improves beyond 11.930 peak; transfer norm_jumpstart may exceed 120%+ on B→C.
  Add to config.yml as `fusion_300k_train_b`, `fusion_300k_transfer_a`, `fusion_300k_transfer_c`.

- [ ] **Train attention_idfree on Env B at 300k steps** — confirm the >100% norm_jumpstart is
  structural (architecture) not a one-run artifact. Both extractors showing it at 300k would be
  the strongest possible paper claim.

### Priority 2 — More difficult transfer: C→A (maximum-gap experiment)

The current B→A transfer does NOT force dc_id renumbering (edge DC keeps id=2 in Env A). This is why
attention_pooling still leads B→A jumpstart despite its dc_id bug. C→A eliminates this loophole.

- [ ] **Train fusion, attention_idfree, euromlsys, attention_pooling on Env C** (300k steps).
  Transfer all four to Env A (50k). Oracle for Env A already exists from prior runs.
  Expected: attention_pooling collapses on C→A (DC IDs 4-5 disappear entirely, their embeddings
  are never used in Env A, plus IDs shift). Fusion and attention_idfree maintain performance.
  This is the experiment that makes the dc_id embedding case watertight for reviewers.

- [ ] **C→B transfer** (same four extractors, Env C source → Env B target at 50k).
  Completes the full transfer matrix.

### Priority 3 — Confirm oracle design (self-normalisation)

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

## Transfer Analysis Insights (2026-05-13)

Results from the full extractor comparison (100k train B, 50k transfer A/C, per-extractor oracle normalization).

### Key Finding 1 — DC ID Embedding Asymmetry

`attention_pooling` (188k params, NO adaptation layer) wins B→A zero-shot (80.6%) and is fastest
to 80% oracle (episode 1). Yet it falls to 81.5% on B→C zero-shot — behind euromlsys (91.3%).

**Mechanism:**
- B→A (remove cloud DC): the absent DC's slot goes to all-zeros. The attention query learned
  "slot 1 = cloud → downweight when empty." The zero-padded slot triggers the learned ignore
  behavior reliably. DC ID embeddings help because the *absent* slots are structurally predictable.
- B→C (add 2 far-edge DCs): DC slots 4-5 were always zero-padded in Env B training. Their
  ID embeddings trained on nothing. When those slots activate in Env C, the query doesn't know
  how to attend to them — the embeddings for those positions are essentially random.

**Publishable claim:** DC positional embeddings create *asymmetric transfer biases* — they
facilitate downscale transfer (empty slots are predictable) but impede upscale transfer
(novel active slots have untrained embeddings).

### Key Finding 2 — Topology Stationarity Hypothesis

Permutation-invariant extractors (spane, hybrid, type_stratified) were designed assuming DC
ordering is arbitrary. That assumption is false for this problem: DC slot 1 is always the
cloud DC, slots 2-3 are edge, slots 4-5 are far-edge (in Env C). The positional structure
is *stationary across environment variants*.

- Permutation invariance discards this stable structure → systematic performance penalty
- euromlsys (flat dense MLP, no symmetry constraint) implicitly learns positional encoding
  for each DC slot → benefits from the stationary structure
- spane (most permutation-invariant) is consistently the worst performer across both directions

**Design principle:** only enforce permutation invariance if the mapping between physical DC
slots and semantic DC roles genuinely changes across environments. In this benchmark it does not.

### Key Finding 3 — Scaling Ablation Is a Strong Negative Result

Scaled body capacity to ~155-160k params (matching euromlsys) for hybrid, hybrid_pre_head,
type_stratified, type_stratified_pre_head. None closed the B→C gap. Best: hybrid_pre_head_scaled
at 90.1% jumpstart vs euromlsys 91.3%, AUC 1.151 vs 1.166.

**Conclusion:** the euromlsys advantage is structural, not from parameter count. The dense
Linear(120→128) first layer can compute cross-host interactions within a single matrix
multiply — it learns "when host 3 in DC 1 is busy AND host 2 in DC 3 is free, prefer DC 3."
Weight-tied per-DC MLPs processing 3 features each cannot express this regardless of hidden_dim.

**Paper value:** clean isolation of structural inductive bias vs capacity. Most reviewers will
assume "bigger = better" — this ablation directly refutes that for this task class.

### Key Finding 4 — Adaptation Layer Placement (Post > Pre)

Pre-head adaptation variants (hybrid_v2, type_stratified_v2) performed WORSE than post-head
originals despite having 15× more adaptation params (65,792 vs 4,192 for hybrid).

**Why:** pre-head adaptation modifies the intermediate representation before the final linear
projection. If the representation is already well-calibrated for the target environment, the
adaptation layer introduces destructive noise. Post-head adaptation modifies the policy output
directly — it's closer to a bias-correction at the action level, which is more stable.

The 0.1× residual scaling (`base + 0.1 * adapt(base)`) encodes the right inductive prior:
"the source policy is mostly correct; the adaptation is a small correction." Pre-head adaptation
with a full Linear(256→256) violates this prior — it can move the representation far from the
source.

### Key Finding 5 — euromlsys Body Provides Cross-Host Interactions

euromlsys body: Linear(120→128) first layer = 15,360 params just for the first projection.
Processes all 120 raw infr features simultaneously, enabling cross-host interactions within
one matrix multiply. Body total = 74,752 params vs 14,048 for type_stratified.

This 5.3× body size advantage is NOT reproducible by scaling the weight-tied MLP. The
architectural difference is qualitative: dense all-pairs vs. per-DC independent processing.

### Key Finding 6 — Fusion + attention_idfree Break the Oracle Ceiling (2026-05-15)

**The headline result of the entire extractor comparison.**

| Extractor | B→C norm jumpstart | B→C norm peak | B→C AUC ratio |
|---|---|---|---|
| fusion | **114.6%** | **129.0%** | **1.441** |
| attention_idfree | **107.9%** | **126.1%** | 1.377 |
| euromlsys (best before) | 91.3% | 103.5% | 1.166 |

Both new extractors have norm_jumpstart > 1.0: at episode 0 of fine-tuning in Env C — before
a single gradient update — they already outperform the oracle's ALL-TIME peak after 50k scratch steps.
Fusion peak reward (15.623) is 29% above the oracle ceiling (12.116).

**Root cause:** Both extractors share two properties no prior extractor had simultaneously:
1. **No dc_id embeddings** — upscale introduces new DC IDs with untrained/random embeddings; removing
   them eliminates the noise floor entirely. Confirmed by comparison: attention_pooling (with dc_id) at
   81.5% vs attention_idfree (without) at 107.9% — a 26pp swing from removing one component.
2. **dc_type embeddings (not raw float)** — each infrastructure tier (cloud/edge/micro) gets an
   independent learned vector instead of ordinal 1<2<3. In Env C with more DCs of each type, the model
   recognizes each DC's role correctly and applies the right strategy.

**B→A downscale picture:** attention still leads jumpstart (80.6%). Fusion (64.7%) and attention_idfree
(57.8%) are mid-table — but both achieve norm_peak > 100% (105.8% and 104.0% respectively), meaning
they eventually EXCEED oracle ceiling there too. The downscale case needs harder experiments (C→A, where
dc_id renumbering is unavoidable) to show where each architecture truly breaks.

**Paper narrative shift:** this is no longer an incremental comparison — it is a demonstration that
cross-environment transfer can produce policies strictly better than training from scratch in the target,
given the right representation. This is the headline claim reviewers will remember.

### Scientific Contributions for Journal Paper

1. **Architecture:** PPO-X custom extractor with residual adaptation for cross-domain transfer
2. **Empirical evaluation:** 6+ extractors × 2 transfer directions × per-extractor oracle
   normalization — most comprehensive transfer evaluation in cloud-edge DRL
3. **Topology stationarity principle:** when DC roles are fixed, permutation invariance hurts
4. **Capacity ablation:** structural advantage is independent of parameter count
5. **DC ID asymmetry:** positional embeddings create directional transfer bias (downscale ✓ / upscale ✗)
6. **Adaptation placement:** post-head bounded residual is better than pre-head full adaptation
7. **Zero-shot super-oracle transfer:** fusion + attention_idfree achieve >100% oracle ceiling
   at zero-shot — the clearest empirical case for transfer learning value in this domain class.

### What to implement next (ordered by expected signal/effort)

1. **RBF-attention** (done 2026-05-13) — replaces MHA in hybrid with Gaussian kernel pooling.
   Tests whether sparse content-based weights outperform near-uniform MHA on ≤8 DC tokens.
2. **hybrid + DC-type conditioned query** — give each DC type its own learned query vector
   (nn.Embedding(max_dc_types+1, dc_emb_dim)) instead of one global query. Bridges hybrid
   and type_stratified: type-aware WITHOUT positional DC ID bias. Medium effort.
3. **Contrastive pre-training** (Idea C) — highest effort, highest novelty. Pre-train on
   topology augmentations (DC shuffle, DC drop, host zero) to explicitly optimize invariance.

## Next Feature Extractors to Try

### Architecture ideas (in priority order)

- [ ] **Idea A — PMA multi-query attention pooling** (no dc_type label dependency)
  Use `max_dc_types` separate learned query vectors (one per slot) instead of one global query.
  Each query attends over all DC tokens independently and specialises on its own cluster.
  This is the PMA (Pooling by Multihead Attention) block from Set Transformer (Lee et al., 2019).
  Advantage over `type_stratified`: doesn't require dc_type labels to be correct; queries learn groupings from load patterns alone.
  Watch for query degeneration — add orthogonality regularisation if queries collapse.
  New file: `common/rl-manager/extractors/pma_extractor.py`

- [ ] **Idea B — Hierarchical job–DC cross-attention**
  Current extractors encode jobs and DCs independently, merging only at the head via concatenation.
  A cross-attention step (jobs as queries, DC type embeddings as keys/values) lets each job's representation ask "given my cores/deadline, how does this infrastructure look?"
  Produces per-job placement scores rather than a single global infrastructure summary.
  Larger architectural change — best built on top of `type_stratified` or PMA.

- [ ] **Idea C — Contrastive topology invariance (SimCLR-style pre-training)**
  Pre-train the extractor with a contrastive objective: two augmented views of the same topology
  (shuffle DC order, add/remove a DC of the same type, randomly zero a host) should be close in
  embedding space; different topologies should be far apart.
  Explicitly optimises for the invariance that transfer learning requires.
  Training protocol: pre-train on Env B rollouts with contrastive loss only → fine-tune end-to-end with RL.
  New pieces needed: augmentation functions for obs dicts, pre-training loop outside SB3.
  Estimated effort: 1–2 weeks.

### 2026 Transformer literature to explore for inspiration

Papers that claim to fix fundamental Transformer limitations — potentially relevant for the DC pooling / job-attention components:

- [ ] **Softmax Linear Attention** (Xu et al., 2026) — `arXiv:2602.01744`
  Addresses why prior linear-attention fixes (Linformer, Performer) lost global competition.
  Uses inter-head softmax gates on linear backbones → O(n) with winner-take-all dynamics preserved.
  Relevance: could replace the MHA pooling step in `hybrid`/`hybrid_pre_head` with a cheaper but equally expressive pooling.

- [ ] **Scaling Attention via Feature Sparsity (SFA)** (Xie et al., ICLR 2026) — `arXiv:2603.22300`
  Represents queries and keys as k-sparse codes; custom FlashSFA kernel reduces cost from Θ(n²d) to Θ(n²k²/d).
  Relevance: our DC token sets are small (≤8 tokens), so quadratic cost is not the bottleneck — but the sparse-code representation idea could make DC embeddings more interpretable/transferable.

- [ ] **Linear Transformer: ReLU-based linear attention** (Zhu, 2026) — DOI:10.1117/12.3116015
  Reorders computation into query-independent KV aggregation + per-query readout. Strict O(n) runtime.
  Relevance: cleanest drop-in for the `pool_attn` step in the hybrid extractor.

- [ ] **Hyperloop Transformers** (Zeitoun et al., 2026) — `arXiv:2604.21254`
  Looped middle block with hyper-connections (matrix-valued residual streams). 50% fewer params at equal quality.
  Relevance: parameter-efficiency framing aligns with the "match euromlsys at equal param count" question.

- [ ] **OOD generalisation via latent space reasoning** (ICLR 2026) — openreview.net/forum?id=Wjgq9ISdP0

- [ ] **OLS is a Special Case of Transformer** (arXiv:2604.13656) — *background theory only, no implementation*
  Proves single-layer linear transformer = OLS closed-form projection. Reveals "fast memory" (in-context KV) vs "slow memory" (weights) structure. Useful for understanding *why* linear attention approximations (SLA, ReLU-linear) work — contextualises the papers above but doesn't suggest new architecture.

- [ ] **RBF-Attention** — github.com/4rtemi5/rbf_attention — *implement as drop-in for pool_attn in hybrid*
  Replaces dot-product similarity with a Gaussian RBF kernel: sim(q,k) = exp(-‖q−k‖²/σ²). Naturally bounded [0,1], no softmax needed, bandwidth σ controls selectivity. Well-suited to small token sets (≤8 DCs) where "find DCs with similar capacity profile" is the right semantic rather than global ranking. Simple drop-in: replace `nn.MultiheadAttention` in `hybrid_extractor.py` with an RBF-weighted sum over DC tokens.

- [ ] **SWAT: Structure-Aware Transformer for Inhomogeneous Multi-Task RL** — openreview.net/forum?id=fy_XRVHqly — *read carefully before next extractor*
  Directly addresses *inhomogeneous multi-task RL* where tasks have different topology sizes — exactly the B/A/C problem. Uses structure metadata to condition the transformer on the current task's topology rather than relying purely on invariant representations. Could explain why pure invariance approaches (spane, type_stratified) under-perform: they discard topological structure that is actually informative.

### Related work to cite (not implement)

- **CAT: Cross-domain Adaptive Transfer RL** (PMLR 2022) — proceedings.mlr.press/v180/you22a.html
  Learns explicit state-action correspondence mappings between source and target envs. Strong citation for the transfer methodology section of the TPDS paper.

- **Snowflake: Scaling GNNs via Parameter Freezing** (arXiv:2103.01009)
  Freeze base GNN params, only fine-tune adapter during transfer across different robot morphologies. Direct citation support for the `freeze_inactive_input_layer_weights` ablation (P2 item 5 in plan).

- **NerveNet** (ICLR 2018) — historical background; TURRET already cites and adapts this.
  Four mechanisms for OOD-length generalisation: input-adaptive recurrence, algorithmic supervision,
  anchored discrete latents, explicit error correction.
  Relevance: the B→C upscale transfer IS an OOD problem (more DCs, different mix). Anchored discrete
  latent representations could complement the type-stratified approach.

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

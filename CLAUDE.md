# CLAUDE.md

CloudSim-RL: Reinforcement learning training system for cloud resource allocation. Bridges **Java CloudSim Plus** (discrete-event simulator) with **Python Stable-Baselines3** (RL agent) via **gRPC**, running inside Docker.

---

## Behavioral Guidelines

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop and ask before touching code.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- No error handling for impossible scenarios.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused, not pre-existing ones.

Every changed line should trace directly to the user's request.

### Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan before starting:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

### Feature Extractor Safety

**Before creating or reviewing any feature extractor, read the transfer-safety checklist.**

The checklist lives at: `~/.claude/projects/-home-taslanidis-git-rl-cloudsimplus/memory/extractor_transfer_checklist.md`

It documents 4 architectural red flags that cause silent degradation on cross-environment transfer and are easy to overlook:
1. DC ID embeddings (reviewer-fatal)
2. Raw dc_type scalar into linear layer (ordinal assumption)
3. Unmasked mean pooling over DC slots (count-shift between envs)
4. Positional flat MLP without canonical ordering guarantee

Run the quick audit greps from the checklist before claiming any extractor is transfer-safe.

---

## Project Structure

```
rl-cloudsimplus/
├── Makefile                          # Root — sets domain=, includes common/Makefile
├── common/
│   ├── Makefile                      # All make targets (build, run, clean, proto, etc.)
│   ├── docker-compose.yml            # Manager service with volume mounts
│   ├── versions.gradle               # Single source of truth: manager/gateway/gradle versions
│   ├── proto/unified/
│   │   └── cloudsimplus.proto        # Canonical proto — edit only here
│   ├── cloudsimplus-gateway-shared/  # Shared Java module (base classes, interfaces)
│   │   ├── gradlew                   # Single Gradle wrapper for all domain builds
│   │   └── src/main/java/daislab/cspg/
│   ├── rl-manager/
│   │   ├── Dockerfile
│   │   ├── startup.sh                # pip install -e at container start
│   │   ├── gym_cloudsimplus/         # Python gRPC client + gymnasium envs (volume mounted)
│   │   ├── utils/misc.py             # Java spawning, config parsing, RL helpers
│   │   ├── train.py / transfer.py / test.py
│   │   ├── entrypoint.py             # Dispatcher: reads config, calls train/transfer/test
│   │   └── callbacks/
│   └── scripts/
│       └── run_docker.sh             # Multi-experiment sequencing
├── domain/
│   ├── vm-management/                # Single-DC VM lifecycle RL
│   │   ├── config.yml
│   │   ├── topologies/               # YAML datacenter topology definitions
│   │   ├── traces/                   # Job CSV trace files
│   │   ├── cloudsimplus-gateway/     # Domain Java source (extends shared)
│   │   └── rl-manager/entrypoint.py
│   └── job-placement/                # Multi-DC job-to-datacenter RL
│       ├── config.yml
│       ├── topologies/
│       ├── traces/
│       ├── cloudsimplus-gateway/
│       └── rl-manager/entrypoint.py
└── docs/
    └── architecture.md               # Extended architecture notes
```

---

## Build Commands

```bash
# Full build (Docker image + Java JAR)
make build domain=<domain>

# Java gateway JAR only (after Java code changes)
make build-gateway domain=<domain>

# Docker manager image only (rarely needed separately)
make build-manager domain=<domain>

# Run experiment(s) defined in config.yml
make run domain=<domain>

# Start TensorBoard at http://localhost:6006 (no domain needed)
make run-tensorboard

# Kill the experiment orchestrator, then stop containers
make stop-all

# Full cleanup: containers, images, gradle build, logs
make clean-all domain=<domain>

# Wipe logs only
make wipe-logs domain=<domain>

# Regenerate Python pb2 classes from canonical proto
make generate-proto

# Show built JAR version
make check-gateway-jar domain=<domain>

# Show CloudSim Plus dependency version
make check-gateway-deps domain=<domain>
```

`domain=` is required for all targets. Supported values: `vm-management`, `job-placement`.

---

## Code Change Workflow

| Change | Action required |
|--------|----------------|
| Python (`gym_cloudsimplus/`, `utils/`, `entrypoint.py`) | **None** — volume-mounted, reinstalled at container start |
| Java (`cloudsimplus-gateway/`) | `make build-gateway domain=<domain>` |
| Proto (`common/proto/unified/cloudsimplus.proto`) | `make generate-proto` → `make build-gateway` for both domains |
| `config.yml` | **None** — read at runtime |
| Java log level / destination | **None** — env vars at runtime |

**Proto change workflow** — only needed when editing the `.proto` file:
```bash
make generate-proto                        # regenerates Python pb2 classes
make build-gateway domain=vm-management    # Java copies proto automatically
make build-gateway domain=job-placement
make run domain=<domain>
```

**Never `cd` into gateway directories** to build. Always use `make build-gateway`.

---

## Architecture

### End-to-End Flow

```
config.yml
    │
    ▼
entrypoint.py ──► train.py / transfer.py / test.py
                        │
                        ▼
                  SubprocVecEnv  (N parallel workers)
                        │
               ┌────────┴────────┐
               │  Python worker  │  (×N)
               │  CloudSimBaseEnv│
               │  ─────────────  │
               │  CloudSimGrpc   │
               │  Client         │
               └───────┬─────────┘
                       │ gRPC (localhost:5005x)
                       ▼
               Java JVM subprocess
               CloudSimGrpcService
                       │
               CloudSimProxy (domain)
                       │
               CloudSim Plus simulation
```

- **Python** spawns one Java JVM per worker via `subprocess.Popen` inside `misc.py:spawn_java_gateway()`
- Each JVM listens on its own port (`base_port + worker_rank`)
- **gRPC RPCs**: `createSimulation`, `reset`, `step`, `batchStep`, `close`, `ping`
- The Java JAR is mounted read-only into the container; Python code is mounted live

### Config Flow

```
config.yml
  globals:  ──► run_docker.sh reads → env vars → docker-compose → container env
  common:   ──┐
              ├─► dict_from_config() merges → params dict → Java via JSON string
  experiment_N:┘                                           → Python directly
```

---

## Java Architecture

### Shared vs Domain Classes

All domain-agnostic logic lives in `common/cloudsimplus-gateway-shared/`. Domain packages only contain what differs.

**Shared classes** (`common/cloudsimplus-gateway-shared/src/main/java/daislab/cspg/`):

| Class | Role |
|-------|------|
| `Main` | JVM entry point — parses args, configures logback, starts gRPC server |
| `GrpcServer` | Wraps io.grpc server lifecycle |
| `CloudSimGrpcServiceBase` | gRPC RPC implementations (createSimulation, reset, step, batchStep, close, ping) |
| `CloudSimProxyBase` | Simulation stepping engine: clock management, job queue, event loop |
| `WrappedSimulationBase` | Bridges proxy ↔ gRPC service; implements `IWrappedSimulation`; `reset()` concrete, `step()` abstract |
| `SimulationFactoryBase` | Parses JSON params, builds `ISimulationSettings`, creates the simulation |
| `ISimulationSettings` | Interface for settings shared by both domains + static parsing helpers |
| `ICloudSimProxy` | Interface: `runOneTimestep()`, `isRunning()`, `reset()`, `terminate()` |
| `IWrappedSimulation` | Interface: `step()`, `reset()`, `getObservation()` |
| `Observation` | Value object: `int[] infrastructureObservation`, `int[] secondaryObservation` |
| `GrpcServiceDelegate` | Static helpers for proto ↔ Java conversion |
| `CloudletDescriptor` | Parsed job entry from CSV trace |
| `DatacenterBrokerFirstFitFixed` | Custom broker with fixed first-fit VM selection |
| `VmAllocationPolicyCustom` | RL-controlled VM allocation (no-op; Python drives decisions) |
| `OptimizedCloudletScheduler` | Cloudlet scheduler optimized for simulation throughput |
| `TreeArray` | Utility: serializes DC→Host→VM→Job hierarchy to flat int array |

**Domain classes** (in `domain/<domain>/cloudsimplus-gateway/src/main/java/daislab/cspg/`):

| Class | vm-management | job-placement |
|-------|:---:|:---:|
| `SimulationSettings` | ✓ | ✓ |
| `SimulationFactory` | ✓ | ✓ |
| `CloudSimGrpcService` | ✓ | ✓ |
| `CloudSimProxy` | ✓ | ✓ |
| `WrappedSimulation` | ✓ | ✓ |
| `HostWithoutCreatedList` | ✓ | — |
| `DatacenterWithType` | — | ✓ |
| `CloudletWithLocation` / `CloudletDescriptorWithLocation` | — | ✓ |

### Design Patterns

**Template Method** — `CloudSimProxyBase.runOneTimestep()` defines the fixed sequence (clear lists → submit jobs → advance clock → print stats). Abstract hooks let domains customize: `setupInfrastructure()`, `tryToSubmitJobs()`, `getPrimaryDatacenter()`, `printStats()`.

**Factory** — `SimulationFactoryBase.create()` is the single entry point for constructing a simulation from a JSON params string. Domain `SimulationFactory` overrides to instantiate domain-specific `SimulationSettings` and `WrappedSimulation`.

**Value Object** — `Observation`, `SimulationStepResult`, `SimulationResetResult` are Lombok `@Value` / immutable records — never mutated after construction.

### Observation Schema

`Observation` carries two `int[]` arrays sent over proto:

| Field | vm-management | job-placement |
|-------|---------------|---------------|
| `infrastructureObservation` | Tree-array: DC→Host→VM→Job hierarchy (flat encoding) | `[dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts]` per host, all DCs; dc_type 1=cloud, 2=edge, 3=micro (0 = padding) |
| `secondaryObservation` | `[min(jobCoresWaiting, maxVmPes)]` (length 1) | `[cores, location, nominal_runtime_ref, time_to_due, s0, s1, s2]` per visible job: the first `max_jobs_waiting` arrived jobs by (due, arrival, id); action[i] places slot i |

`HOST_OBS_FEATURES = 5` (`WrappedSimulation` / `JobPlacementEnv`) and the 7-feature job wire format (`CloudSimProxy.JOB_OBS_FEATURES` / `JobPlacementEnv._JOB_WIRE_FEATURES`) **must** agree on both sides; `tests/test_observation_schema.py` checks it. Python strips `location` from the policy's job features and turns it into the `reach_mask` channel.

### Key Invariants

- **Java system properties must come before `-jar`**: `-Dlog.level=INFO -jar gateway.jar` — not after.
- `firstStep` flag: the first timestep uses `timestepInterval` as target time, not `clock() + interval`, because the clock starts at `minTimeBetweenEvents` not 0.
- `VmAllocationPolicyCustom` is a no-op allocator — the RL agent drives placement via action decoding; CloudSim Plus itself never decides where to place VMs in RL mode.

**Job-placement reward (net SLA value, `SlaLedger`):**
- Each job resolves exactly once: `V − c` if it finishes by `arrival + deadline`, else `−P − c`; `c = cost_<tier> · pes · length / mips_ref`, 0 if never placed. A step's reward is that step's resolutions divided by `Z = ΣV`, so placing nothing scores exactly `−ΣP/ΣV`.
- Every action path (RL and both heuristics) binds through `WrappedSimulation.bind()`, so all policies are scored by the same ledger.
- Unplaced jobs past their due time are evicted from the queue. At `max_episode_length` the remaining unplaced jobs are violated and the simulation drains until every placed job resolves, so episodes always terminate (never truncate).
- A full DC stays a legal action: the VM selector ignores free PEs and the job queues. Python's mask is `reach ∧ (DC has a VM with ≥ cores PEs)`; keep the two in lockstep.
- Network delay counts from binding (in `tryToSubmitJobs`); jobs still in flight count as used PEs and backlog in the observation.
- The benchmark metric is the sum of `info["unshaped_reward"]`. With `reward_shaping: true` the reward adds `γΦ' − Φ`, and `reward_shaping_gamma` must equal the algorithm's γ (checked in train/transfer).

---

## Python Architecture

### Gym Environment Hierarchy

```
CloudSimBaseEnv (envs/base.py)          ← abstract base; gRPC wiring, reset(), step(), close()
├── VmManagementEnv (envs/vm_management.py)
└── JobPlacementEnv (envs/job_placement.py)
```

**`CloudSimBaseEnv`** provides:
- `_client: CloudSimGrpcClient` — gRPC stub
- `reset()`, `step()`, `close()`, `ping()`
- `_pad_observation(obs, target_dim)` — zero-pads to fixed shape; raises if the observation is longer
- Abstract methods: `_get_observation()`, `_parse_step_info()`, `action_masks()`

**`VmManagementEnv`**:
- Action space: `MultiDiscrete([3, max_hosts, max_vms, vm_types_count])` — `[action_type, host_id, vm_id, vm_type]`
- Obs space: `Dict(infr_state: Box, job_cores_waiting_state: Box(1,))`
- Derives `vm_cores`, `host_count`, `host_pes` from topology params — never hardcoded
- Maintains `host_cores_utilized` and `vms_running` for action masking

**`JobPlacementEnv`**:
- Action space: `MultiDiscrete([max_datacenters] × max_jobs_waiting)`
- Obs space: `Dict(infrastructure_state: Box, jobs_waiting_state: Box, reach_mask: Box)` — 960 + 192 + 768 with the benchmark shape constants (`total_hosts` 192, `max_jobs_waiting` 32, `max_datacenters` 24, pinned in config)
- `reach_mask[j, k]` = topology allows placing job slot j via action k; the action mask is reach ∧ free PEs, and a padding slot can only take the no-op

### gRPC Client

`CloudSimGrpcClient` (`cloud_sim_grpc_client.py`) wraps all RPC calls. Key method:

```python
_obs_to_dict(obs) → {"infrastructure_observation": [...], "secondary_observation": [...]}
```

Both domains use the same observation dict keys — the split between a single scalar vs. an array is transparent at this layer.

### Training Pipeline

```
entrypoint.py
  │  reads globals → injects save_experiment into params
  │  reads experiment params → preprocesses DCs/jobs (jp only)
  │
  ▼
train.py / transfer.py / test.py
  │  creates SubprocVecEnv or DummyVecEnv
  │  each worker: spawn_java_gateway() → env.__init__() → createSimulation gRPC
  │
  ▼
SB3 algorithm (PPO, MaskablePPO, A2C, …)
  │  rollout: env.step(action) ←→ Java step gRPC
  ▼
callbacks: SaveOnBestTrainingRewardCallback
loggers: stdout + CSV + TensorBoard (if save_experiment=true)
```

---

## Config System

Each `config.yml` has three sections:

### `globals:` — container-level settings
Read by `run_docker.sh` and `entrypoint.py`. Passed as environment variables.

| Key | Values | Notes |
|-----|--------|-------|
| `attached` | `true/false` | Attach terminal to container output |
| `gpu` | `true/false` | Enable CUDA profile |
| `java_log_level` | `TRACE\|DEBUG\|INFO\|WARNING\|ERROR` | Logback level |
| `java_log_destination` | comma-separated `stdout`, `file`, or `none` | e.g. `stdout,file` — order-independent |
| `save_experiment` | `true/false` | Gates Python CSV/TB logging **and** Java file logging |
| `num_cpu` | integer | Number of parallel simulation workers |

### `common:` — shared experiment parameters
Merged into every experiment's params dict.

### `experiment_N:` — per-experiment overrides
Keys: `mode` (`train`/`transfer`/`test`), `experiment_dir`, `experiment_name`, `datacenters`, `job_trace_filename`, `train_model_dir` (for transfer/test).

**Run directories are write-once.** `log_dir` = `base_log_dir/experiment_dir/experiment_name`
is created with no `exist_ok`, so a rerun can never merge into a previous run's output.

| target state | behaviour |
|--------------|-----------|
| absent / empty | runs normally |
| a run that crashed or was killed | archived automatically, no flag |
| a completed or pre-`run_status.json` run | needs `on_exists: archive`, else the experiment is **skipped** (exit 3) and the queue continues |

`on_exists` accepts only `abort` (default) and `archive` — there is deliberately no
`overwrite`; use `rm -rf` for that. Archives land in `base_log_dir/_archive/<experiment_dir>/<experiment_name>/<UTC stamp>/`
with their `tfevents` files renamed to `tfarchived` so TensorBoard ignores them.
Clear them with `make prune-archive`.

`make run` runs `common/scripts/preflight.py` first, which prints the whole queue with
resolved sources and predicted skips. It blocks only when a transfer/test source model
neither exists on disk nor is produced by an earlier experiment in the same queue.

**`base_log_dir` is the single source of truth for where results land.** `run_docker.sh`
reads it once from `common:` and exports it as `BASE_LOG_DIR`, which drives all three
consumers so they cannot drift apart:

| consumer | resolves to |
|----------|-------------|
| container bind mount (`docker-compose.yml`) | `../common/${BASE_LOG_DIR}:/mgr/${BASE_LOG_DIR}` |
| host path preflight inspects | `common/$BASE_LOG_DIR` |
| Python (`entrypoint`, `transfer`, `test`, `run_dir`) | `params["base_log_dir"]` |

It must be set once under `common:` — a per-experiment override is rejected by preflight,
because the bind mount is established once per container and could not honour it.
The two domain-free Make targets (`run-tensorboard`, `prune-archive`) cannot read a
per-domain config; they use `LOGS_DIR ?= common/logs`, so pass `LOGS_DIR=...` if you change it.

### Policy & algorithm params

Each domain has its own controlling-policy param. `rl_algorithm` is read **only** when the policy is `rl`.

**vm-management:**

| Key | Values | Behavior |
|-----|--------|----------|
| `vm_allocation_policy` | `rl` | RL agent decides VM lifecycle (create/destroy) AND host placement explicitly. Uses `VmAllocationPolicyCustom`. |
| | `fromfile` | Replays recorded actions from a file. Uses `VmAllocationPolicyCustom`. |
| | `minimize-queue` | Rule: create a VM if no running VM has enough free cores; otherwise destroy the largest idle VM. |
| | `minimize-allocated` | Rule: scan VM types largest→smallest; create/destroy to keep allocated cores tight. |
| | `minimize-unutilized` | Rule: scan VM types smallest→largest; create only the type that exactly matches a waiting job. |
| `rl_algorithm` | any SB3/sb3-contrib algo (e.g. `PPO`, `MaskablePPO`) | Read only when `vm_allocation_policy: rl`. |

For rule-based modes, host placement is delegated to `VmAllocationPolicyBestFit` under the hood — the rule decides *what* and *when*, bestfit decides *where*.

**job-placement:** see [Two-Stage Placement Model](#two-stage-placement-model) below.

### Two-Stage Placement Model

Job placement is a **two-stage decision pipeline**, mirroring how real cloud orchestration systems work (Kubernetes: cluster autoscaler → scheduler; AWS: region selection → AZ/instance):

| Stage | Decision | Config key | Options |
|-------|----------|-----------|---------|
| 1 (macro) | Which **datacenter** runs this cloudlet? | `cloudlet_to_dc_mapping` | `rl`, `earliest-shortest-to-most-free-dc`, `earliest-most-critical-to-nearest-dc` |
| 2 (micro) | Which **VM within that DC** runs the cloudlet? | `cloudlet_to_vm_mapping` | `most-free-pes` (currently the only option; pluggable) |

The RL agent operates on **stage 1 only** (when `cloudlet_to_dc_mapping: rl`). Stage 2 is always rule-based — a tactical decision better handled by a simple rule than learning.

CloudSim Plus has no native "cloudlet → DC" concept (cloudlets bind to VMs via `bindCloudletToVm`); this two-stage model is a higher-level abstraction layered on top, with VM selection happening inside the chosen DC. VM-to-host placement is hardcoded to bestfit in job-placement (not configurable — the RL agent never controls it).

`rl_algorithm` is read only when `cloudlet_to_dc_mapping: rl`.

---

## Environment Variables (Java subprocess)

Set from `globals:` via `run_docker.sh` → docker-compose → `entrypoint.py` → JVM `-D` properties:

| Env var | JVM property | Default |
|---------|-------------|---------|
| `JAVA_LOG_LEVEL` | `log.level` | `INFO` |
| `JAVA_LOG_DESTINATION` | `log.destination` | `stdout` |
| `SAVE_EXPERIMENT` | `log.saveExperiment` | `true` |
| `JAVA_SIM_LOG_DIR` | `log.simDir` | (empty → `logs/`) |
| `EXPERIMENT_ID` | `experiment.id` | `default` |

Java file logging only activates when **both** `log.saveExperiment=true` and `log.destination` contains `file`.

---

## Build System Details

- **Single Gradle wrapper**: `common/cloudsimplus-gateway-shared/gradlew` — builds both domains as subprojects (`:vm-management`, `:job-placement`)
- **Proto copy**: on each `make build-gateway`, the canonical proto is copied to `domain/<domain>/cloudsimplus-gateway/src/main/proto/` before Gradle runs
- **Versions**: `common/versions.gradle` defines `managerVersion`, `gatewayVersion`, `gradleVersion` — referenced by both Makefile and domain `build.gradle` files
- **`-PuseSnapshot=true`** is the default for `make build-gateway`; pass `ARGS=-PuseSnapshot=false` to override

---

## Code Style

### Java
- Java 21+, Lombok (`@Value` for immutable records, `@Data` for mutable settings)
- 4-space indentation
- Named constants over magic numbers — e.g. `JOB_OBS_FEATURES = 7` instead of literal `7`
- `@SuppressWarnings("unchecked")` on methods with necessary raw casts (data from JSON deserialization), not scattered inline
- Domain-agnostic logic belongs in shared module; only domain-specific behaviour in domain packages

### Python
- Type hints on public methods
- No class-level hardcoded numeric constants — derive from config params or named class variables
- Gym env `__init__` must call `super().__init__()` before domain setup
- `_pad_observation()` lives in the base class — do not copy it to subclasses

---

## Adding a New Domain

1. Create `domain/<name>/` with `config.yml`, `topologies/`, `traces/`, `cloudsimplus-gateway/`, `rl-manager/entrypoint.py`
2. In Java: extend `CloudSimProxyBase`, `WrappedSimulationBase`, `SimulationFactoryBase`, `CloudSimGrpcServiceBase`; implement `step()` (including the reward), `extractInfrastructureObservation()`, `extractSecondaryObservation()`
3. In Python: extend `CloudSimBaseEnv`; implement `_get_observation()`, `_parse_step_info()`, `action_masks()`
4. Add the domain to `common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/cloud_sim_grpc_client.py`
5. Register environment in `gym_cloudsimplus/__init__.py`
6. Add Gradle subproject entry in `common/cloudsimplus-gateway-shared/settings.gradle`

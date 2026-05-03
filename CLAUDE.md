# CLAUDE.md

CloudSim-RL: Reinforcement learning training system for cloud resource allocation. Bridges **Java CloudSim Plus** (discrete-event simulator) with **Python Stable-Baselines3** (RL agent) via **gRPC**, running inside Docker.

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

# Start TensorBoard at http://localhost:6006
make run-tensorboard domain=<domain>

# Stop containers
make stop

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
| `WrappedSimulationBase` | Bridges proxy ↔ gRPC service; assembles observations via strategy objects |
| `SimulationFactoryBase` | Parses JSON params, builds `ISimulationSettings`, creates the simulation |
| `ISimulationSettings` | Interface for settings shared by both domains + static parsing helpers |
| `ICloudSimProxy` | Interface: `runOneTimestep()`, `isRunning()`, `reset()`, `terminate()` |
| `IWrappedSimulation` | Interface: `step()`, `reset()`, `getObservation()` |
| `IStateExtractor` | Strategy: `extractState()` → `int[]` infrastructure observation |
| `IActionDecoder` | Strategy: decodes RL action array → simulation action |
| `IRewardCalculator` | Strategy: computes scalar reward from step result |
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
| `VmManagementStateExtractor` / `JobPlacementStateExtractor` | ✓ | ✓ |
| `VmManagementActionDecoder` / `JobPlacementActionDecoder` | ✓ | ✓ |
| `VmManagementRewardCalculator` / `JobPlacementRewardCalculator` | ✓ | ✓ |
| `VmCost` | ✓ | — |
| `HostWithoutCreatedList` | ✓ | — |
| `DatacenterWithType` | — | ✓ |
| `CloudletWithLocation` / `CloudletDescriptorWithLocation` | — | ✓ |

### Design Patterns

**Template Method** — `CloudSimProxyBase.runOneTimestep()` defines the fixed sequence (clear lists → submit jobs → advance clock → print stats). Abstract hooks let domains customize: `setupInfrastructure()`, `tryToSubmitJobs()`, `getPrimaryDatacenter()`, `printStats()`.

**Strategy** — `IStateExtractor`, `IActionDecoder`, `IRewardCalculator` are injected into `WrappedSimulationBase`. Swapping them changes domain behavior without touching the stepping engine.

**Factory** — `SimulationFactoryBase.create()` is the single entry point for constructing a simulation from a JSON params string. Domain `SimulationFactory` overrides to wire domain-specific strategies.

**Value Object** — `Observation`, `SimulationStepResult`, `SimulationResetResult` are Lombok `@Value` / immutable records — never mutated after construction.

### Observation Schema

`Observation` carries two `int[]` arrays sent over proto:

| Field | vm-management | job-placement |
|-------|---------------|---------------|
| `infrastructureObservation` | Tree-array: DC→Host→VM→Job hierarchy (flat encoding) | `[dc_id, dc_type_id, free_vmpes]` per host, all DCs |
| `secondaryObservation` | `[min(jobCoresWaiting, maxVmPes)]` (length 1) | `[cores, location, sensitivity, deadline]` per waiting job (length = `JOB_OBS_FEATURES × jobs`) |

`JOB_OBS_FEATURES = 4` is a named constant in `CloudSimProxy` (Java) and `JobPlacementEnv` (Python). Both sides **must** agree on this value.

### Key Invariants

- **Java system properties must come before `-jar`**: `-Dlog.level=INFO -jar gateway.jar` — not after.
- `firstStep` flag: the first timestep uses `timestepInterval` as target time, not `clock() + interval`, because the clock starts at `minTimeBetweenEvents` not 0.
- `VmAllocationPolicyCustom` is a no-op allocator — the RL agent drives placement via action decoding; CloudSim Plus itself never decides where to place VMs in RL mode.

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
- `_pad_observation(obs, target_dim)` — zero-pads or truncates to fixed shape
- Abstract methods: `_get_observation()`, `_parse_step_info()`, `action_masks()`

**`VmManagementEnv`**:
- Action space: `MultiDiscrete([3, max_hosts, max_vms, vm_types_count])` — `[action_type, host_id, vm_id, vm_type]`
- Obs space: `Dict(infr_state: Box, job_cores_waiting_state: Box(1,))`
- Derives `vm_cores`, `host_count`, `host_pes` from topology params — never hardcoded
- Maintains `host_cores_utilized` and `vms_running` for action masking

**`JobPlacementEnv`**:
- Action space: `MultiDiscrete([max_datacenters] × max_jobs_waiting)`
- Obs space: `Dict(infrastructure_state: Box, jobs_waiting_state: Box)`
- `JOB_OBS_FEATURES = 4` class constant (must match Java)

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
- Named constants over magic numbers — e.g. `JOB_OBS_FEATURES = 4` instead of literal `4`
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
2. In Java: extend `CloudSimProxyBase`, `WrappedSimulationBase`, `SimulationFactoryBase`, `CloudSimGrpcServiceBase`; implement `IStateExtractor`, `IActionDecoder`, `IRewardCalculator`
3. In Python: extend `CloudSimBaseEnv`; implement `_get_observation()`, `_parse_step_info()`, `action_masks()`
4. Add the domain to `common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/cloud_sim_grpc_client.py`
5. Register environment in `gym_cloudsimplus/__init__.py`
6. Add Gradle subproject entry in `common/cloudsimplus-gateway-shared/settings.gradle`

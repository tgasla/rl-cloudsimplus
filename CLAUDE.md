# CLAUDE.md

CloudSim-RL: Reinforcement learning training system for cloud resource allocation using CloudSim Plus + Stable-Baselines3 via gRPC.

## Project Structure

```
rl-cloudsimplus/
├── common/
│   ├── Makefile              # Root makefile, delegates to common/Makefile
│   ├── Makefile              # All make targets with paper= argument
│   ├── docker-compose.yml    # Docker services (manager, manager-cuda)
│   ├── rl-manager/
│   │   ├── Dockerfile        # Manager image definition
│   │   ├── startup.sh        # Runtime pip install from volume mounts
│   │   ├── gym_cloudsimplus/ # Python gRPC client (volume mounted)
│   │   ├── utils/            # misc.py (Java spawning, RL helpers)
│   │   ├── train.py          # Training entry point
│   │   └── callbacks/        # SB3 callbacks
│   └── scripts/
│       └── run_docker.sh     # Handles multi-experiment sequencing
├── papers/
│   ├── main/                 # Single-DC architecture
│   │   ├── config.yml
│   │   ├── cloudsimplus-gateway/  # Java gateway source
│   │   └── rl-manager/entrypoint.py
│   └── euromlsys/            # Multi-DC architecture
│       ├── config.yml
│       ├── cloudsimplus-gateway/
│       └── rl-manager/entrypoint.py
└── Makefile                  # Root, sets paper= and includes common/Makefile
```

## Key Files

| File | Purpose |
|------|---------|
| `common/Makefile` | All make targets (build, run, clean, etc.) |
| `common/docker-compose.yml` | Manager service definition with volume mounts |
| `common/rl-manager/startup.sh` | Pip installs gym_cloudsimplus from volume at runtime |
| `common/rl-manager/utils/misc.py` | Java JVM spawning, gRPC env creation, RL helpers |
| `common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/cloud_sim_grpc_client.py` | Paper-aware gRPC client (main vs euromlsys) |
| `common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/envs/multidc.py` | Multi-DC Gymnasium env |
| `common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/protos/` | Separate proto binaries per paper |

## Make Commands

```bash
# Build
make build paper=<paper>          # Build Docker image + Java JAR
make build-gateway paper=<paper>   # Build Java gateway JAR only
make build-manager paper=<paper>   # Build Docker manager image only

# Run
make run paper=<paper>            # Run experiment(s)
make run-tensorboard paper=<paper>  # Start TensorBoard at localhost:6006

# Cleanup
make stop paper=<paper>           # Stop containers, keep images
make clean-all paper=<paper>      # Full cleanup (containers, images, gradle, logs)
make wipe-logs paper=<paper>      # Delete all logs

# Utilities
make check-gateway-jar paper=<paper>   # Show JAR path and version
make check-gateway-deps paper=<paper>   # Show CloudSim Plus version
```

## Code Change Workflow

| Change Type | Action Required |
|-------------|-----------------|
| Python code (gym_cloudsimplus, utils, etc.) | **No rebuild** — just `make run` |
| Java code (cloudsimplus-gateway) | `make build-gateway paper=<paper>` then `make run` |
| Config (config.yml) | **No rebuild** — just `make run` |
| Java log level/destination | **No rebuild** — env vars passed at runtime |

**Important**: Python code is volume-mounted and reinstalled at container startup via `pip install -e`. No Docker rebuild needed for Python changes.

## Build System

Gradle wrapper lives in `common/cloudsimplus-gateway-shared/` — **single source of truth for building both papers' Java gateways**. Papers no longer have their own gradle wrappers.

**Proto file consolidation:**
- **Canonical proto:** `common/proto/unified/cloudsimplus.proto` — edit here to update the proto
- Both papers need local copies in `papers/*/cloudsimplus-gateway/src/main/proto/` for the gradle protobuf plugin
- On every `make build-gateway`, the Makefile copies the canonical proto to the target paper's `src/main/proto/` directory before building

**Build commands only — never cd into gateway directories:**
```bash
make build-gateway paper=main      # copies proto → builds papers/main/cloudsimplus-gateway JAR
make build-gateway paper=euromlsys # copies proto → builds papers/euromlsys/cloudsimplus-gateway JAR
```

**Versions:** `common/versions.gradle` is the single source of truth for managerVersion, gatewayVersion, gradleVersion. Used by both Makefile and papers' build.gradle.

## Papers

The project supports multiple paper architectures:

- `paper=main` — Single-DC RL with VmAllocationPolicy=rl, reward-based metrics
- `paper=euromlsys` — Multi-DC RL with cloudlet_to_dc_assignment_policy=rl, placement ratio metrics

Each paper has its own gRPC client selection in `cloud_sim_grpc_client.py`, and environment classes (`GrpcSingleDC` vs `GrpcMultiDC`). Proto definition lives in `common/proto/unified/cloudsimplus.proto` (single source, copied to papers during build).

## Architecture Notes

- **gRPC communication**: Python spawns Java JVM subprocesses, each with its own gRPC server on a unique port (base_port + worker_rank)
- **Parallel simulation**: SubprocVecEnv spawns multiple Python workers, each running its own Java JVM
- **Volume mounts**: Python code is mounted live; Java JAR is read-only mounted (built once via build-gateway)
- **Paper-aware client**: `CloudSimGrpcClient` selects correct proto based on `paper` parameter
- **Java system properties**: Must come BEFORE `-jar` flag (e.g., `-Dlog.level=INFO -jar ...`)

## Environment Variables (for Java subprocess)

These are read from config.yml globals and passed to Java:

- `JAVA_LOG_DESTINATION` — none|stdout|file|stdout-file
- `JAVA_LOG_LEVEL` — TRACE|DEBUG|INFO|WARNING|ERROR
- `EXPERIMENT_ID` — passed to Java as `-Dexperiment.id`
- `JAVA_SIM_LOG_DIR` — passed as `-Dlog.simDir` for logback-generated.xml location
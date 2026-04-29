# CloudSim-RL

Reinforcement learning training system for cloud resource allocation. Bridges **Java CloudSim Plus** (the simulator) with **Python Stable-Baselines3** (the RL agent) via **gRPC**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Container (manager)             │
│                                                         │
│  ┌─────────────────┐    gRPC     ┌──────────────────┐ │
│  │  Python RL      │◄──────────► │  Java CloudSim    │ │
│  │  Stable-Baselines3 │  JSON   │  Plus Gateway     │ │
│  │  (train/transfer/test) │     │  (Simulation)     │ │
│  └─────────────────┘            └──────────────────┘  │
│         ▲                              ▲               │
│         │         pip install -e       │               │
│  Volume mount (code changes auto-picked up)            │
└─────────────────────────────────────────────────────────┘
```

- **Python side**: RL training with Stable-Baselines3, communicates via gRPC
- **Java side**: CloudSim Plus simulation running in a JVM subprocess
- **Paper support**: `main` (single-DC) and `euromlsys` (multi-DC) share proto definitions from `common/proto/unified/` (copied to papers during build) and use a shared Gradle wrapper at `common/cloudsimplus-gateway-shared/`

---

## Quick Start

```bash
# 1. Build everything from scratch
make clean-all paper=euromlsys
make build paper=euromlsys

# 2. Run experiment
make run paper=euromlsys
```

---

## Make Commands Reference

### Build Commands

| Command | When to Use |
|---------|-------------|
| `make build paper=<paper>` | Build Docker image AND Java gateway JAR. Needed after clean or when infrastructure changes. |
| `make build-gateway paper=<paper>` | Build only the Java gateway JAR (`build/libs/*.jar`). Use when you changed Java code. |
| `make build-manager paper=<paper>` | Build only the Docker manager image. Rarely needed separately. |
| `make build-tensorboard paper=<paper>` | Build the TensorBoard visualization image. |

### Run Commands

| Command | Description |
|---------|-------------|
| `make run paper=<paper>` | Run experiment(s). Reads config from `papers/<paper>/config.yml` |
| `make run-tensorboard paper=<paper>` | Start TensorBoard dashboard at [http://localhost:6006](http://localhost:6006) |

### Cleanup Commands

| Command | Description |
|---------|-------------|
| `make stop` | Stop running containers, remove networks |
| `make clean-all paper=<paper>` | Full cleanup: stop containers, remove images, clean gradle build, wipe logs |
| `make wipe-logs paper=<paper>` | Delete all logs for the paper |
| `make clean-gateway paper=<paper>` | Clean only the Java gradle build |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make check-gateway-deps paper=<paper>` | Show CloudSim Plus dependency version |
| `make check-gateway-jar paper=<paper>` | Show built JAR path and version |
| `make get-gradle paper=<paper>` | Download correct Gradle version for wrapper |
| `make update-cloudsimplus` | Pull latest CloudSim Plus from Git master, publish to local maven |

---

## The `paper=` Argument

The project supports multiple research papers with different configurations:

```bash
paper=main       # Single-DC RL (VmAllocationPolicy=rl), reward-based metrics
paper=euromlsys  # Multi-DC RL (cloudlet_to_dc_assignment_policy=rl), placement ratio metrics
```

Each paper has its own:
- `papers/<paper>/config.yml` — experiment configuration
- `papers/<paper>/topologies/` — datacenter topology definitions
- `papers/<paper>/rl-manager/entrypoint.py` — paper-specific entry point
- `papers/<paper>/cloudsimplus-gateway/` — Java gateway source code
- `papers/<paper>/traces/` — job trace CSV files

**Shared infrastructure (no per-paper copies):**
- Proto definition: `common/proto/unified/cloudsimplus.proto` — single source, copied to papers during build
- Gradle wrapper: `common/cloudsimplus-gateway-shared/gradlew` — used for all gateway builds
- Version definitions: `common/versions.gradle` — single source for all components

The default paper is `main`.

---

## Code Change Workflow

### Python Code Changes
**No rebuild needed.** Python code is volume-mounted into the container and reinstalled at runtime:

```bash
# Just run — volume mounts pick up changes automatically
make run paper=euromlsys
```

The `startup.sh` script runs `pip install --no-deps -e /mgr/gym_cloudsimplus` on container start, installing from the mounted source.

### Java Code Changes
**Rebuild the gateway JAR:**

```bash
make build-gateway paper=euromlsys   # Build JAR
make run paper=euromlsys             # Run (JAR is mounted via :ro volume)
```

### Configuration Changes (config.yml)
**No rebuild needed.** Config is read at runtime:

```bash
# Just run with new config
make run paper=euromlsys
```

### Environment Variables (java_log_level, java_log_destination)
**No rebuild needed.** These are passed via docker-compose environment variables to the Java subprocess at runtime.

---

## Image Volume Mounts

The Docker setup uses volume mounts so code changes take effect without rebuilding:

```yaml
# docker-compose.yml volumes (relevant mounts)
- ./rl-manager/gym_cloudsimplus:/mgr/gym_cloudsimplus    # Python gRPC client (mounted, not copied)
- ./rl-manager/startup.sh:/common/rl-manager/startup.sh   # Startup script
- ../${PAPER_DIR}/rl-manager/entrypoint.py:/mgr/entrypoint.py  # Paper entrypoint
- ../${PAPER_DIR}/config.yml:/mgr/config.yml            # Config file
- ../${PAPER_DIR}/cloudsimplus-gateway/build/libs:/app/cloudsimplus-gateway/build/libs:ro  # JAR (read-only)
```

**Important**: The JAR is mounted `:ro` (read-only) because it's built once via `make build-gateway`. Python code is mounted live so edits to `gym_cloudsimplus/` take effect immediately without rebuild.

---

## Configuration File (config.yml)

Each paper has its own `papers/<paper>/config.yml` with two sections:

### globals (container-level settings)
```yaml
globals:
    attached: true          # true=attach terminal to output, false=run detached
    gpu: false              # true=use CUDA GPU
    java_log_level: INFO    # Java logging level
    java_log_destination: file  # none|stdout|file|stdout-file
    num_cpu: 16             # Number of parallel simulation workers
```

### common (experiment parameters)
Shared across all experiments in the config:
- `seed`: random or integer
- `timesteps`, `n_rollout_steps`: RL training parameters
- `save_experiment`: save model/checkpoints
- `base_log_dir`: parent log directory
- `vm_allocation_policy`: rl | bestfit | rule-based
- `algorithm`: PPO, MaskablePPO, A2C, etc.

### experiment_{id} (per-experiment overrides)
```yaml
experiment_1:
    mode: train              # train | transfer | test
    experiment_dir: euromlsys
    experiment_name: env_a_train
    datacenters: !include topologies/euromlsys_a.yml
    job_trace_filename: euromlsys_jobs_first_50.csv
```

- `mode`: train (new training), transfer (fine-tune from checkpoint), test (eval only)
- When mode=transfer or mode=test, add `train_model_dir` pointing to trained model directory

---

## Experiment Modes

| Mode | Description |
|------|-------------|
| `train` | Train RL agent from scratch |
| `transfer` | Continue training from a saved model (transfer learning) |
| `test` | Evaluate trained model without training |

```yaml
# Train example
experiment_1:
    mode: train
    experiment_dir: euromlsys
    experiment_name: env_a_train

# Transfer example
experiment_2:
    mode: transfer
    experiment_dir: euromlsys
    experiment_name: env_b_transfer
    train_model_dir: euromlsys/env_a_train  # Load from this directory

# Test example
experiment_3:
    mode: test
    experiment_dir: euromlsys
    experiment_name: env_b_eval
    train_model_dir: euromlsys/env_a_train
```

---

## Multi-Experiment Runs

The `config.yml` can define multiple experiments (`experiment_1`, `experiment_2`, etc.). The `run_docker.sh` script detects all experiments and runs them sequentially.

```bash
# With 2 experiments in config.yml, this runs:
# 1. Experiment 1 → cleanup → Experiment 2 → cleanup
make run paper=euromlsys
```

Each experiment:
1. Starts container(s)
2. Runs to completion (or timeout)
3. Cleans up containers
4. Proceeds to next experiment

---

## GPU Support (CUDA)

```bash
# Build with CUDA support
make build paper=euromlsys GPU=true

# Run with GPU
make run paper=euromlsys GPU=true
```

Requirements:
- CUDA installed on host
- `nvidia-container-toolkit` installed
- Edit `/etc/nvidia-container-runtime/config.toml` setting `no-cgroups = false` if GPU not detected
- Restart docker daemon: `sudo systemctl restart docker`

---

## Reproducibility Checklist

### Clean build from scratch:
```bash
make clean-all paper=euromlsys   # Wipe everything
make build paper=euromlsys       # Fresh build
make run paper=euromlsys         # Run
```

### Daily development (Python-only changes):
```bash
make run paper=euromlsys         # No build needed
```

### After Java changes:
```bash
make build-gateway paper=euromlsys  # Rebuild JAR only
make run paper=euromlsys              # Run
```

### Full reset:
```bash
make clean-all paper=euromlsys
make build paper=euromlsys
make run paper=euromlsys
```

---

## Log Output

Training metrics are printed in real-time when `attached: true`:

```
manager  | |    ep_jobs_placed_ratio        | 0.148     |
manager  | |    ep_deadline_violation_ratio | 0.765     |
manager  | |    ep_quality_ratio            | 0.587     |
manager  | |    ep_total_rew                | -0.471    |
```

Logs are saved to `papers/<paper>/rl-manager/logs/<experiment_dir>/<experiment_name>/`.

Java simulation logs (`csp.current.log`) go to the same log directory (controlled by `java_log_destination`).

---

## Troubleshooting

### "No such file or directory" for JAR
The JAR must exist before running. If missing:
```bash
make build-gateway paper=euromlsys
```

### Permission errors on logs/
```bash
make wipe-logs paper=euromlsys   # Wipe then retry
```

### Container build failures
```bash
make clean-all paper=euromlsys   # Full reset
make build paper=euromlsys       # Rebuild
```

### Java gRPC server not starting
Check that the JAR exists at the mounted path. Check container logs for Java exceptions.

---

## Requirements

- **Docker** and **Docker Compose** (no local Java/Python required)
- Optional GPU: CUDA + nvidia-container-toolkit

---

## Acknowledgements

- [CloudSim Plus](http://cloudsimplus.org/) — discrete event simulation framework
- Based on work by [pkoperek](https://github.com/pkoperek):
  - [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway)
  - [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus)
  - [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)
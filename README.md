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
- **Domain support**: `vm-management` (single-DC) and `job-placement` (multi-DC) share proto definitions from `common/proto/unified/` (copied to domains during build) and use a shared Gradle wrapper at `common/cloudsimplus-gateway-shared/`

---

## Quick Start

```bash
# 1. Build everything from scratch
make clean-all domain=job-placement
make build domain=job-placement

# 2. Run experiment
make run domain=job-placement
```

---

## Make Commands Reference

### Build Commands

| Command | When to Use |
|---------|-------------|
| `make build domain=<domain>` | Build Docker image AND Java gateway JAR. Needed after clean or when infrastructure changes. |
| `make build-gateway domain=<domain>` | Build only the Java gateway JAR (`build/libs/*.jar`). Use when you changed Java code. |
| `make build-manager domain=<domain>` | Build only the Docker manager image. Rarely needed separately. |
| `make build-tensorboard domain=<domain>` | Build the TensorBoard visualization image. |

### Run Commands

| Command | Description |
|---------|-------------|
| `make run domain=<domain>` | Run experiment(s). Reads config from `domain/<domain>/config.yml` |
| `make run-tensorboard domain=<domain>` | Start TensorBoard dashboard at [http://localhost:6006](http://localhost:6006) |

### Cleanup Commands

| Command | Description |
|---------|-------------|
| `make stop` | Stop running containers, remove networks |
| `make clean-all domain=<domain>` | Full cleanup: stop containers, remove images, clean gradle build, wipe logs |
| `make wipe-logs domain=<domain>` | Delete all logs for the domain |
| `make clean-gateway domain=<domain>` | Clean only the Java gradle build |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make check-gateway-deps domain=<domain>` | Show CloudSim Plus dependency version |
| `make check-gateway-jar domain=<domain>` | Show built JAR path and version |
| `make get-gradle domain=<domain>` | Download correct Gradle version for wrapper |
| `make update-cloudsimplus` | Pull latest CloudSim Plus from Git master, publish to local maven |

---

## The `domain=` Argument

The project supports multiple RL problem domains:

```bash
domain=vm-management  # Single-DC RL (VmAllocationPolicy=rl), reward-based metrics
domain=job-placement   # Multi-DC RL (cloudlet_to_dc_assignment_policy=rl), placement ratio metrics
```

Each domain has its own:
- `domain/<domain>/config.yml` — experiment configuration
- `domain/<domain>/topologies/` — datacenter topology definitions
- `domain/<domain>/rl-manager/entrypoint.py` — domain-specific entry point
- `domain/<domain>/cloudsimplus-gateway/` — Java gateway source code
- `domain/<domain>/traces/` — job trace CSV files

**Shared infrastructure (no per-domain copies):**
- Proto definition: `common/proto/unified/cloudsimplus.proto` — single source, copied to domains during build
- Gradle wrapper: `common/cloudsimplus-gateway-shared/gradlew` — used for all gateway builds
- Version definitions: `common/versions.gradle` — single source for all components

The default domain is `vm-management`.

---

## Code Change Workflow

### Python Code Changes
**No rebuild needed.** Python code is volume-mounted into the container and reinstalled at runtime:

```bash
# Just run — volume mounts pick up changes automatically
make run domain=job-placement
```

The `startup.sh` script runs `pip install --no-deps -e /mgr/gym_cloudsimplus` on container start, installing from the mounted source.

### Java Code Changes
**Rebuild the gateway JAR:**

```bash
make build-gateway domain=job-placement   # Build JAR
make run domain=job-placement             # Run (JAR is mounted via :ro volume)
```

### Configuration Changes (config.yml)
**No rebuild needed.** Config is read at runtime:

```bash
# Just run with new config
make run domain=job-placement
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
- ../${DOMAIN_DIR}/rl-manager/entrypoint.py:/mgr/entrypoint.py  # Domain entrypoint
- ../${DOMAIN_DIR}/config.yml:/mgr/config.yml            # Config file
- ../${DOMAIN_DIR}/cloudsimplus-gateway/build/libs:/app/cloudsimplus-gateway/build/libs:ro  # JAR (read-only)
```

**Important**: The JAR is mounted `:ro` (read-only) because it's built once via `make build-gateway`. Python code is mounted live so edits to `gym_cloudsimplus/` take effect immediately without rebuild.

---

## Configuration File (config.yml)

Each domain has its own `domain/<domain>/config.yml` with two sections:

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
    experiment_dir: job_placement
    experiment_name: env_a_train
    datacenters: !include topologies/job_placement_a.yml
    job_trace_filename: job_placement_jobs_first_50.csv
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
    experiment_dir: job_placement
    experiment_name: env_a_train

# Transfer example
experiment_2:
    mode: transfer
    experiment_dir: job_placement
    experiment_name: env_b_transfer
    train_model_dir: job_placement/env_a_train  # Load from this directory

# Test example
experiment_3:
    mode: test
    experiment_dir: job_placement
    experiment_name: env_b_eval
    train_model_dir: job_placement/env_a_train
```

---

## Multi-Experiment Runs

The `config.yml` can define multiple experiments (`experiment_1`, `experiment_2`, etc.). The `run_docker.sh` script detects all experiments and runs them sequentially.

```bash
# With 2 experiments in config.yml, this runs:
# 1. Experiment 1 → cleanup → Experiment 2 → cleanup
make run domain=job-placement
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
make build domain=job-placement GPU=true

# Run with GPU
make run domain=job-placement GPU=true
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
make clean-all domain=job-placement   # Wipe everything
make build domain=job-placement       # Fresh build
make run domain=job-placement         # Run
```

### Daily development (Python-only changes):
```bash
make run domain=job-placement         # No build needed
```

### After Java changes:
```bash
make build-gateway domain=job-placement  # Rebuild JAR only
make run domain=job-placement              # Run
```

### Full reset:
```bash
make clean-all domain=job-placement
make build domain=job-placement
make run domain=job-placement
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

Logs are saved to `domain/<domain>/rl-manager/logs/<experiment_dir>/<experiment_name>/`.

Java simulation logs (`csp.current.log`) go to the same log directory (controlled by `java_log_destination`).

---

## Troubleshooting

### "No such file or directory" for JAR
The JAR must exist before running. If missing:
```bash
make build-gateway domain=job-placement
```

### Permission errors on logs/
```bash
make wipe-logs domain=job-placement   # Wipe then retry
```

### Container build failures
```bash
make clean-all domain=job-placement   # Full reset
make build domain=job-placement       # Rebuild
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
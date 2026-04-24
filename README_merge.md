# CloudSim-RL

Reinforcement learning training system for cloud resource allocation. Bridges **Java CloudSim Plus** (the simulator) with **Python Stable-Baselines3** (the RL agent) via **gRPC**.

## Quick Start

```bash
# 1. Copy and edit config
cp config.template.yml config.yml

# 2. Build everything (Java JAR + Docker images)
make build

# 3. Run experiments
make run
```

## Requirements

- Docker + Docker Compose
- No local Java/Python required — both run inside containers

## Configuration

Edit `config.yml`. Key global settings:

| Key | Values | Description |
|-----|--------|-------------|
| `attached` | `true` / `false` | Attach terminal to experiment output |
| `gpu` | `true` / `false` | Use CUDA GPU |
| `java_log_level` | `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR` | Java logging verbosity |
| `java_log_destination` | `stdout`, `file`, `stdout-file`, `none` | Where Java logs go |
| `junit_output_show` | `true` / `false` | Show JUnit output during build |

Java log files are written to `logs/experiment_${EXPERIMENT_ID}/cspg.current.log`, one per experiment.

## Architecture

Each experiment runs as a separate Docker container. Inside the container, Python spawns 16 Java JVM subprocesses (one per worker). Each JVM runs a CloudSim simulation and communicates with Python via gRPC on its own port (50051–50066).

Python → DummyVecEnv (sequential) → GrpcSingleDC → gRPC → Java CloudSim

## Makefile Targets

```bash
make build          # Build Java JAR + Docker images
make build-gateway  # Build Java gateway only
make build-manager  # Build Docker manager image only
make run            # Run experiments from config.yml
make run-tensorboard # Start TensorBoard dashboard
make stop           # Stop containers and prune volumes
make clean-gateway  # Clean Gradle build outputs
make wipe-logs      # Clear logs directory
```

## Version Management

All version strings are in `versions.gradle` — update it there, then rebuild.

## Development

### Hot Reload

Source code is mounted as volumes inside the container:
- `rl-manager/mnt/` → `/mnt` (Python code)
- `cloudsimplus-gateway/build/libs/` → `/app/cloudsimplus-gateway/build/libs` (Java JAR, read-only)

Changes to Python code take effect immediately. Java code changes require `make build-gateway` to rebuild the JAR, then the new JAR is visible inside the container.

### Debugging Java

Set `java_log_level: DEBUG` and `java_log_destination: stdout` in config.yml to see all Java log output.

## Acknowledgements

- [CloudSim Plus](http://cloudsimplus.org/) — Java 17+ cloud simulation framework
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — RL library
- Based on work by [pkoperek](https://github.com/pkoperek): [cloudsimplus-gateway](https://github.com/pkoperek/cloudsimplus-gateway), [gym_cloudsimplus](https://github.com/pkoperek/gym_cloudsimplus), [dqn_cloudsimplus](https://github.com/pkoperek/dqn_cloudsimplus)

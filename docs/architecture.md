# Project Architecture

## Overview

CloudSim-RL is a reinforcement learning training system for cloud resource allocation. It bridges **Java CloudSim Plus** (the simulator) with **Python Stable-Baselines3** (the RL agent) via **gRPC**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (manager container)                │
│  Stable-Baselines3 ──► DummyVecEnv ──► gRPC client (16x)   │
│                                        (one per worker)      │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         │         gRPC (TCP localhost)       │
         │                                    │
         ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│               Java (16x separate JVM subprocesses)          │
│  GrpcServer ──► CloudSimGrpcService ──► WrappedSimulation   │
│        one per JVM, ports 50051-50066                        │
│  CloudSim event-driven simulation                            │
└─────────────────────────────────────────────────────────────┘
```

## Mode: gRPC (`use_grpc: true`)

Python spawns **16 Java JVMs** as subprocesses (one per `num_cpu`). Each JVM runs a `GrpcServer` on its own port (50051–50066). Python uses `DummyVecEnv` to wrap 16 `GrpcSingleDC` env instances, each connected to one JVM via gRPC.

**Critical limitation**: `DummyVecEnv` runs envs **sequentially** — only 1 JVM is ever active at a time. The 16 Java JVMs sit mostly idle while Python steps through them one by one. This is because gRPC channels cannot be pickled for `SubprocVecEnv` IPC.

## Mode: Py4J (`use_grpc: false`)

Single JVM, Py4J gateway server. Python imports Java objects directly. **Not used / deprecated.**

## Directory Structure

```
rl-cloudsimplus/
├── config.yml                          # Main configuration file
├── config.template.yml                 # Config template
├── docker-compose.yml                  # Docker services: manager, gateway (py4j)
│
├── cloudsimplus-gateway/               # Java CloudSim gRPC server
│   ├── src/main/java/daislab/cspg/
│   │   ├── Main.java                  # Entry point (gRPC vs Py4J dispatcher)
│   │   ├── GrpcServer.java            # gRPC Netty server, one per JVM
│   │   ├── CloudSimGrpcService.java   # gRPC service impl (create/reset/step/close)
│   │   ├── SimulationFactory.java     # Creates WrappedSimulation from params
│   │   ├── WrappedSimulation.java     # Main simulation wrapper (step, reset)
│   │   ├── SimulationSettings.java    # Configuration bean (from params map)
│   │   ├── SimulationStepInfo.java   # Step metadata (rewards, tree array, etc.)
│   │   ├── SimulationResetResult.java
│   │   ├── SimulationStepResult.java
│   │   ├── Observation.java           # Domain object (infrastructure obs)
│   │   ├── CloudSimProxy.java         # CloudSim abstraction layer
│   │   ├── CloudletDescriptor.java    # Job descriptor (JSON serde)
│   │   ├── DatacenterBrokerFirstFitFixed.java
│   │   ├── HostWithoutCreatedList.java
│   │   ├── OptimizedCloudletScheduler.java
│   │   ├── VmAllocationPolicyCustom.java
│   │   ├── VmCost.java
│   │   ├── MultiSimulationEnvironment.java  # [Py4J only]
│   │   ├── SimulationHistory.java          # [Py4J only]
│   │   ├── MetricsStorage.java             # [Py4J only]
│   │   ├── TimeMeasurement.java            # [Py4J only]
│   │   └── TreeArray.java                  # [Dead code]
│   └── src/main/proto/
│       └── cloudsimplus.proto             # gRPC/Protobuf schema
│
├── rl-manager/                        # Python RL training
│   ├── mnt/
│   │   ├── entrypoint.py             # Docker entrypoint
│   │   ├── train.py                  # Main training loop
│   │   ├── transfer.py               # Transfer learning
│   │   ├── utils/
│   │   │   └── misc.py               # Env factory, DummyVecEnv setup, callback creation
│   │   └── callbacks/
│   │       └── save_on_best_training_reward_callback.py
│   ├── gym_cloudsimplus/             # [Py4J only, deprecated]
│   └── grpc_cloudsimplus/             # gRPC Python client + gym env
│       └── grpc_cloudsimplus/
│           ├── grpc_client.py         # gRPC stub wrapper
│           └── envs/
│               └── grpc_singledc.py   # gym.Env (reset/step)
│
└── scripts/
    └── run_docker.sh                 # Docker launch script (exports UID/GID)
```

## Key Data Flows

### Training Step (`step`)

```
Python: model.learn() → DummyVecEnv.env[rank].step(action)
  └─► GrpcSingleDC.step() → gRPC StepRequest → Java: CloudSimGrpcService.step()
        ├─► WrappedSimulation.step(action)
        │     ├─► executeCustomAction() — create/destroy VMs
        │     ├─► cloudSimProxy.runOneTimestep() — advance simulation
        │     ├─► getInfrastructureObservation() — [O(hosts+VMs+cloudlets)]
        │     ├─► calculateReward()
        │     └─► return SimulationStepResult (obs, reward, done, info)
        └─► gRPC StepResponse → Python: (obs, reward, done, info)
  └─► RL agent: policy.evaluate_actions() — single-threaded Python
```

### Observation Space (RL)

```
RL agent observes:
  obs["infr_state"]        — int array, size 1 + max_hosts + max_vms + max_jobs
                             compact fixed-size encoding, NOT the tree
  obs["job_cores_waiting"] — int, cores of waiting jobs

info dict (debug only, NOT used by RL agent):
  info["observation_tree_array"]  — hierarchical tree of hosts→VMs→cloudlets
  info["job_wait_reward"]        — reward component
  info["unutilized_vm_core_ratio"]
  info["host_affected"]          — which host was affected by last action
  info["cores_changed"]          — cores changed by last action
```

### Configuration Key: `send_observation_tree_array`

- **Java**: `SimulationSettings.getSendObservationTreeArray()` gates whether the tree is computed for `SimulationStepInfo`. The tree for `Observation` (in `infr_state`) is **always computed** — it's the RL agent's state.
- **Python**: `save_on_best_training_reward_callback` gates whether tree arrays are appended/written to CSV files.
- **Proto**: `repeated int32 observation_tree_array = 8` is always sent. Making it optional would save the gRPC transfer overhead when disabled.

## Performance Characteristics

| Component | Time | Notes |
|---|---|---|
| Java CloudSim simulation | ~0.1–0.5ms/step | Event-driven, fast |
| gRPC roundtrip (Python→Java→Python) | ~1–5ms/step | Blocking, sequential |
| DummyVecEnv sequential execution | — | Only 1 of 16 JVMs active per step |
| `getInfrastructureObservation()` | O(hosts+VMs+cloudlets) | Called twice per step |

**Bottleneck**: DummyVecEnv sequential execution. 16 Java JVMs are spawned but only 1 is ever used at a time because Python steps through them sequentially in a single thread.

## Build

```bash
# Build Java gateway (outputs shadow JAR to build/libs/cloudsimplus-gateway-0.1.0.jar)
cd cloudsimplus-gateway && ./gradlew shadowJar

# Full build (JAR + Docker image)
make build-gateway
```

## Dead Code

Files in `cloudsimplus-gateway/` that are **never loaded** in gRPC mode:

- `MultiSimulationEnvironment.java` — Py4J gateway, unused in gRPC
- `SimulationHistory.java` — referenced only by Py4J code
- `MetricsStorage.java` — referenced only by Py4J code
- `TimeMeasurement.java` — referenced only by Py4J code
- `TreeArray.java` — unreferenced utility class, never instantiated
- `py4j` dependency in build.gradle — only needed for Py4J mode

Files in `rl-manager/` that are **never loaded** when `use_grpc: true`:

- `gym_cloudsimplus/` — entire directory, Py4J-only gym env

---

*Last updated: 2026-04-14*

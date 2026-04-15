# TODO

## Completed

- [x] **Fix `send_observation_tree_array` flag propagation** — flag was set in config but never reached the Python callback, so CSV files were always written. Fixed: flag now flows config.yml → train.py/transfer.py → create_callback → callback constructor → guards all write/append sites.
- [x] **Guard tree array computation in Java** — when flag=false, SimulationStepInfo no longer calls `getInfrastructureObservation()` (was computing full tree then throwing it away). RL's `infr_state` (Observation object) still always computed — correct behavior.
- [x] **Use `shadowJar` for fat JAR** — `./gradlew shadowJar` not `jar`; `make build-gateway` now works correctly.

## Optimizations

- [ ] **Guard `getInfrastructureObservation()` — NOT DONE: double-call still exists**: `step()` still calls `getInfrastructureObservation()` twice per step (once for SimulationStepInfo, once for Observation). Refactor to compute once and pass to both. Low priority — RL obs must always compute it.

- [ ] **Batch gRPC calls** — send multiple steps per roundtrip to reduce roundtrip frequency (16x fewer roundtrips). Requires changes to proto schema and both client/server.

- [ ] **Async gRPC client + thread pool** — use async Python gRPC with a thread pool to overlap gRPC wait times across workers. Keeps architecture, no proto changes needed.

- [ ] **SubprocVecEnv with post-fork gRPC** — establish gRPC channels after `fork()`. Risky/complex due to Channel non-picklability.

- [ ] **Optional proto field** — make `observation_tree_array` optional in proto so Java can skip sending entirely when flag=false. Requires proto recompile on both sides.

## FPS / Performance (as of 2026-04-14)

### Benchmark data (user-provided):
| Config | Start FPS | End FPS | Wall time to 4096 steps |
|---|---|---|---|
| 1 CPU | 77 | 34 | 1.53 min |
| 16 CPU | 207 | ~35 | 1.17 min |

**Key finding**: FPS degradation happens even with **1 CPU** (77→34). This means:
- The degradation is NOT caused by DummyVecEnv queue buildup
- The 16-CPU parallelism helps overall (~24% faster) but the degradation curve is similar
- Both converge to ~34-35 FPS regardless of parallelism

**Current hypothesis**: CloudSim event-driven simulation complexity changes over episode lifetime — as the simulation state evolves, per-step cost increases. Or: policy update time grows as model trains.

### Potential fixes for FPS (ordered by effort):
1. **Batch gRPC calls** — easiest: send N steps per roundtrip, reduces gRPC overhead 16x
2. **Async gRPC + threading** — async Python client with thread pool to overlap waits
3. **SubprocVecEnv post-fork** — true Python parallelism
4. **In-process Java via JNI** — no network, fastest but most invasive

### Config observations:
- `num_cpu: 1` gives same final FPS as `num_cpu: 16` — confirms parallelism isn't the bottleneck
- `send_observation_tree_array: false` — tree skip helps CPU but not the fundamental bottleneck

## Architecture Notes

- **16 JVMs**: each spawned as subprocess by `_create_grpc_env_for_rank()` in `misc.py`, each running `GrpcServer` on ports 50051-50066
- **DummyVecEnv**: `vectorize_env()` uses `DummyVecEnv` for gRPC because gRPC Channel can't be pickled for SubprocVecEnv IPC
- **No sleep() in step path**: Java CloudSim simulation itself is fast; gRPC serialization + sequential Python execution is the bottleneck
- **Callback**: `SaveOnBestTrainingRewardCallback._save_timestep_details()` appends to 8 lists every step, plus `load_results()` reads CSV on every episode end — minor overhead

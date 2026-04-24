# Plan: Fix Observation Tree Array Pass-Through

## Context

Despite `send_observation_tree_array: true` being set in `config.yml`, tree arrays are still being passed from Java to Python via gRPC and written to CSV files when the flag is `false`. The Java-side fix (sending dummy `int[1]`) is already implemented but insufficient — the Python side still reads and writes the data to CSV regardless of the flag value.

**Root cause**: The `send_observation_tree_array` flag is read from `params` in Java and controls whether Java sends real tree vs dummy `int[1]`. However, the Python callback `SaveOnBestTrainingRewardCallback`:
1. Always appends `infos[0]["observation_tree_array"]` to `self.observation_tree_arrays` (line 125-127)
2. Always writes to `observation_tree_arrays.csv` on episode end (line 292-294)
3. Always writes to `best_obs_tree_arrays.csv` when new best is found (line 322-324)
4. The flag is never propagated from `params` to the callback constructor

## Scope

Python-side only (Java side already done).

## Why `grpc_client.py:97` Is Out of Scope

**Line 97**: `"observation_tree_array": list(info.observation_tree_array),`

The gRPC client unconditionally extracts `observation_tree_array` into the `info` dict. This was considered but rejected for two reasons:

1. **Requires Java-side proto/schema changes**: The `observation_tree_array` field is defined in the protobuf schema. To conditionally exclude it from the gRPC response, Java would need to either (a) modify the proto definition to make the field optional, or (b) maintain a second proto message variant. Both are outside the Python-only scope of this plan.

2. **Overhead is negligible**: When the flag is `false`, Java sends a dummy `int[1]` (sentinel value `[0]`, confirmed in `SimulationStepInfo.java:42`). Python deserializes this single-element int array on every step. At ~1 step/second for 10k steps, the cost is ~10k small allocations — measurable but not significant in a training workload where the dominant costs are model inference and experience replay.

3. **Alternative considered and rejected**: Adding a Python-side conditional in `grpc_client.py` to gate the extraction based on a cached flag would require propagating the flag into the gRPC client layer and threading it through multiple call sites. This adds coupling between layers without benefit, since the real fix (guarding CSV writes) is already in place.

**Optional hardening (LOW priority)**: A defensive sentinel check could verify that the extracted `int[1]` is not `[0]` (the sentinel value Java sends when the flag is `false`). However, this check would fire on every step when disabled, adding complexity for no functional benefit — CSV writes are already blocked when the flag is `false`.

## Task Flow

### Step 1: Propagate `send_observation_tree_array` from `params` to `create_callback`

**File**: `rl-manager/mnt/utils/misc.py`

Modify `create_callback` to accept the `send_observation_tree_array` flag from params and pass it to `SaveOnBestTrainingRewardCallback`.

- Line 49: Change `def create_callback(save_experiment, log_dir):` to `def create_callback(save_experiment, log_dir, send_observation_tree_array=True):`
- Line 50-51: Pass `send_observation_tree_array` to the `SaveOnBestTrainingRewardCallback` constructor

**Acceptance criteria**: `create_callback` signature accepts `send_observation_tree_array` and forwards it.

### Step 2: Update `train.py` call site to pass the flag

**File**: `rl-manager/mnt/train.py`

- Line 62: Change `create_callback(params["save_experiment"], params["log_dir"])` to include `params.get("send_observation_tree_array", True)`

**Acceptance criteria**: `train.py` passes the flag to `create_callback`.

### Step 3: Update `transfer.py` call site to pass the flag

**File**: `rl-manager/mnt/transfer.py`

- Line 66: Change `create_callback(params["save_experiment"], params["log_dir"])` to include `params.get("send_observation_tree_array", True)`

**Acceptance criteria**: `transfer.py` passes the flag to `create_callback`.

### Step 4: Modify `SaveOnBestTrainingRewardCallback` to respect the flag

**File**: `rl-manager/mnt/callbacks/save_on_best_training_reward_callback.py`

Four places need modification:

1. **Constructor (line 21-30)**: Add `send_observation_tree_array=True` parameter and store as `self.send_observation_tree_array`
2. **`_clear_episode_details` (line 69-82)**: Only append to `self.observation_tree_arrays` when `self.send_observation_tree_array` is `True`
3. **`_on_step` (line 278-327)**: Guard the two `_write_observation_tree_arrays_to_file` calls (lines 292-294 and 322-324) with the flag
4. **`_write_observation_tree_arrays_to_file` (line 271-276)**: Guard the write operation — the method can still exist but do nothing when disabled

**Acceptance criteria**:
- When `send_observation_tree_array=False`, `self.observation_tree_arrays` is never populated and no tree array CSV files are created
- When `send_observation_tree_array=True`, behavior is unchanged
- The `info` dict still receives `observation_tree_array` from `grpc_client.py:97` — this is harmless as the agent never uses it and it is debug-only in `info`

## Detailed TODOs

| # | File | Lines | Change |
|---|------|-------|--------|
| 1 | `rl-manager/mnt/utils/misc.py` | 49-51 | Add `send_observation_tree_array` param to `create_callback`, pass to callback constructor |
| 2 | `rl-manager/mnt/train.py` | 62 | Pass `params.get("send_observation_tree_array", True)` to `create_callback` |
| 3 | `rl-manager/mnt/transfer.py` | 66 | Pass `params.get("send_observation_tree_array", True)` to `create_callback` |
| 4 | `rl-manager/mnt/callbacks/save_on_best_training_reward_callback.py` | 21-30 | Add constructor param `send_observation_tree_array`, store as `self.send_observation_tree_array` |
| 5 | `rl-manager/mnt/callbacks/save_on_best_training_reward_callback.py` | 125-127 | Wrap `self.observation_tree_arrays.append(...)` with `if self.send_observation_tree_array:` |
| 6 | `rl-manager/mnt/callbacks/save_on_best_training_reward_callback.py` | 292-294 | Guard `_write_observation_tree_arrays_to_file("observation_tree_arrays.csv", "a")` with flag |
| 7 | `rl-manager/mnt/callbacks/save_on_best_training_reward_callback.py` | 322-324 | Guard `_write_observation_tree_arrays_to_file("best_obs_tree_arrays.csv", "w")` with flag |

## Success Criteria

1. With `send_observation_tree_array: false` in `config.yml`, no `observation_tree_arrays.csv` or `best_obs_tree_arrays.csv` files are created during training
2. With `send_observation_tree_array: true` in `config.yml`, behavior is identical to current (CSV files created normally)
3. The gRPC client still extracts `observation_tree_array` into the `info` dict (harmless, not used by agent) — this is unchanged
4. All callers of `create_callback` in `train.py` and `transfer.py` pass the flag

## Notes

- `grpc_client.py:97` will continue to include `observation_tree_array` in the `info` dict. This is **intentional** — the field is debug-only and the RL agent never reads it (confirmed: agent uses only `infr_state` and `job_cores_waiting_state`).
- The sentinel value Java sends when the flag is `false` is `int[1]` (default `[0]`, confirmed in `SimulationStepInfo.java:42`). No defensive check is planned for this.
- The flag name in Python code (`send_observation_tree_array`) matches the config key `send_observation_tree_array` in `config.yml:70` for consistency.
- Default value is `True` to preserve backward compatibility if the key is absent from older config files.

## Pre-Mortem (3 Failure Scenarios)

**Scenario 1: Flag not propagated to `transfer.py` call site**
- If `transfer.py` is not updated, transfer learning runs will still write CSV files even when `send_observation_tree_array: false`.
- Detection: Manual inspection of both call sites; automated test catches this if added.
- Mitigation: Include `transfer.py` in the task list and verify both files.

**Scenario 2: CSV files pre-exist from prior runs with flag `false`**
- If CSV files were created in a prior run with `send_observation_tree_array: true`, they are not cleaned up when the flag is later set to `false`.
- Detection: Manual inspection of output directory before a new run.
- Mitigation: Document that users should manually delete existing CSV files when switching the flag from `true` to `false`.

**Scenario 3: Backward compatibility broken for configs without the flag**
- If a user has an old `config.yml` without `send_observation_tree_array`, default `True` preserves old behavior.
- Detection: Existing configs tested manually; CI if added.
- Mitigation: Default is `True`, so absent key = old behavior. Only concern is if someone expected absent key to mean `false`.

## Test Plan

### Unit Tests
- `test_create_callback_respects_flag`: Instantiate `create_callback` with `send_observation_tree_array=False` and verify the callback stores `self.send_observation_tree_array = False`
- `test_callback_no_append_when_disabled`: Call `_clear_episode_details` on a callback with flag `false` and verify `observation_tree_arrays` remains empty

### Integration Tests
- Run a short training job with `send_observation_tree_array: false` and confirm no CSV files appear in `log_dir`
- Run a short training job with `send_observation_tree_array: true` and confirm CSV files are created normally

### E2E Tests
- Full Docker Compose stack: set `send_observation_tree_array: false` in config.yml, run training for 1 episode, verify `observation_tree_arrays.csv` and `best_obs_tree_arrays.csv` do not exist in the log directory
- Flip flag back to `true`, run another episode, verify CSV files are created

### Observability
- Add a log line in `_write_observation_tree_arrays_to_file` when the flag is `false` and the method is called (defensive, fires only if guard is bypassed): `logger.debug("Skipping observation_tree_array CSV write (flag disabled)")`
- The gRPC client deserializing `int[1]` on every step when disabled is visible in Python profiler output but not instrumented further (low priority)

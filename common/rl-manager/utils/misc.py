import re
import os
from copy import deepcopy
import yaml
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)
from callbacks.best_on_val_callback import BestOnValCallback
from utils.evaluation import eval_env_params, level_quotas
from utils import levels
from utils.levels import LevelSampler, LevelSource, SENSITIVITY_LEVELS
from utils.rl_algorithm_support_flags import (
    ALGORITHMS_WITH_ENT_COEF,
    ALGORITHMS_WITH_ACTION_NOISE,
    ALGORITHMS_WITH_N_STEPS,
)


# ─── Gateway version resolution ─────────────────────────────────────────────

def _gateway_version():
    """Read gatewayVersion from versions.gradle at runtime."""
    try:
        with open("/mgr/versions.gradle") as f:
            match = re.search(r"gatewayVersion\s*=\s*['\"]([^'\"]+)['\"]", f.read())
            return match.group(1) if match else "0.1.0"
    except FileNotFoundError:
        return "0.1.0"


# ─── Config parsing ─────────────────────────────────────────────────────────

def _include_constructor(loader, node):
    filename = os.path.join(
        os.path.dirname(loader.stream.name), loader.construct_scalar(node)
    )
    with open(filename, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class _Datacenter:
    def __init__(self, name, dc_type, amount, hosts, connect_to):
        self.name = name
        self.dc_type = dc_type
        self.amount = amount
        self.hosts = hosts if isinstance(hosts, list) else [hosts] if hosts else []
        self.connect_to = connect_to if isinstance(connect_to, list) else [connect_to] if connect_to else []

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.dc_type,
            "amount": self.amount,
            "hosts": [h.to_dict() if hasattr(h, 'to_dict') else h for h in self.hosts],
            "connect_to": self.connect_to,
        }


class _Host:
    def __init__(self, amount, pes, pe_mips, ram, storage, bw, vms):
        self.amount = amount
        self.pes = pes
        self.pe_mips = pe_mips
        self.ram = ram
        self.storage = storage
        self.bw = bw
        self.vms = vms if isinstance(vms, list) else [vms] if vms else []

    def to_dict(self):
        return {
            "amount": self.amount, "pes": self.pes, "pe_mips": self.pe_mips,
            "ram": self.ram, "storage": self.storage, "bw": self.bw,
            "vms": [v.to_dict() if hasattr(v, 'to_dict') else v for v in self.vms],
        }


class _Vm:
    def __init__(self, amount, pes, pe_mips, ram, size, bw):
        self.amount = amount
        self.pes = pes
        self.pe_mips = pe_mips
        self.ram = ram
        self.size = size
        self.bw = bw

    def to_dict(self):
        return {
            "amount": self.amount, "pes": self.pes, "pe_mips": self.pe_mips,
            "ram": self.ram, "size": self.size, "bw": self.bw,
        }


def _datacenter_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return _Datacenter(
        name=fields["name"], dc_type=fields["type"], amount=fields["amount"],
        hosts=fields.get("hosts", []), connect_to=fields.get("connect_to", []),
    )


def _host_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return _Host(
        amount=fields["amount"], pes=fields["pes"], pe_mips=fields["pe_mips"],
        ram=fields["ram"], storage=fields["storage"], bw=fields["bw"],
        vms=fields.get("vms", []),
    )


def _vm_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return _Vm(
        amount=fields["amount"], pes=fields["pes"], pe_mips=fields["pe_mips"],
        ram=fields["ram"], size=fields["size"], bw=fields["bw"],
    )


def _register_yaml_constructors():
    yaml.add_constructor("!include", _include_constructor)
    yaml.add_constructor("!datacenter", _datacenter_constructor)
    yaml.add_constructor("!host", _host_constructor)
    yaml.add_constructor("!vm", _vm_constructor)


def dict_from_config(experiment_id, config):
    _register_yaml_constructors()
    with open(config, "r") as file:
        cfg = yaml.load(file, Loader=yaml.Loader)
    params = {**cfg["common"], **cfg["experiments"][experiment_id - 1]}
    return params


# ─── Datacenter translation helpers (for euromlsys job_placement) ───────────

_sensitivity_mapping = SENSITIVITY_LEVELS


def _get_sensitivity_level(sensitivity_str: str) -> int:
    return _sensitivity_mapping[sensitivity_str]


def _get_dc_idx_by_name(name: str, datacenters: list[dict]) -> int:
    for idx, dc in enumerate(datacenters):
        if dc["name"] == name:
            return idx
    raise ValueError(f"Datacenter with name {name} not found in datacenters list.")


def _check_datacenters_unique(datacenters: list[dict]) -> None:
    names = [dc["name"] for dc in datacenters]
    if len(names) != len(set(names)):
        raise ValueError("Datacenters must have unique names.")


def _check_datacenter_amounts_are_one(datacenters: list[dict]) -> None:
    # Python indexes datacenters by YAML position but Java expands `amount` into several DCs,
    # which would desynchronise the action mask from the simulator.
    multi = [dc["name"] for dc in datacenters if int(dc.get("amount", 1)) != 1]
    if multi:
        raise ValueError(f"datacenter amount must be 1 in job-placement, got >1 for {multi}")


def _translate_connect_to_names_to_idx(datacenters: list[dict]) -> list[dict]:
    for dc in datacenters:
        connect_to_idx = [_get_dc_idx_by_name(c, datacenters) for c in dc.get("connect_to", [])]
        dc["connect_to"] = connect_to_idx
    return datacenters


def _translate_job_location_names_to_idx(jobs: list[dict], datacenters: list) -> list[dict]:
    for job in jobs:
        job["location"] = _get_dc_idx_by_name(job["location"], datacenters)
    return jobs


def _translate_sensitivity_str_to_levels(jobs: list[dict]) -> list[dict]:
    for job in jobs:
        job["delaySensitivity"] = _get_sensitivity_level(job["delaySensitivity"])
    return jobs


# ─── Logger & callback ───────────────────────────────────────────────────────

def create_logger(save_experiment, log_dir) -> sb3.common.logger.Logger:
    log_destination = ["stdout"]
    if save_experiment:
        log_destination.extend(["csv", "tensorboard"])
    return configure(log_dir, log_destination)


def create_callback(save_experiment, log_dir) -> SaveOnBestTrainingRewardCallback | None:
    if save_experiment:
        return SaveOnBestTrainingRewardCallback(log_dir)
    return None


def create_val_callback(params: dict, num_cpu: int) -> tuple[BestOnValCallback, object]:
    """RING-N runs pick checkpoints on the val split instead of the training curve.
    Returns the callback and the val env it plays on; close that env after learn()."""
    workers = params.get("val_num_cpu", 8)
    eval_env = vectorize_env(None, None, num_cpu=workers, jobs_json="[]",
                             params=eval_env_params(params, "val", workers, port_offset=num_cpu))
    callback = BestOnValCallback(eval_env, level_quotas("val", workers), params["log_dir"],
                                 params.get("val_every", 100_000), verbose=1)
    return callback, eval_env


def apply_finetune_scope(model, scope: str) -> None:
    """Which part of a loaded policy keeps training: `full` (default), `head` (everything
    after the features extractor: SB3's MLP/action/value nets, or the pointer head and critic)
    or `extractor` (only the features extractor)."""
    if scope == "full":
        return
    if scope not in ("head", "extractor"):
        raise ValueError(f"finetune must be full, head or extractor, got {scope!r}")
    policy = model.policy
    extractor = {id(p) for module in (policy.features_extractor, policy.pi_features_extractor,
                                      policy.vf_features_extractor) for p in module.parameters()}
    for p in policy.parameters():
        p.requires_grad_((id(p) in extractor) == (scope == "extractor"))


def source_checkpoint(params: dict) -> str:
    """The saved model a transfer/test/evaluate run loads from train_model_dir: `checkpoint`
    if set, else best_val_model for RING-N runs and the legacy best_model otherwise.
    common/scripts/preflight.py applies the same rule."""
    return params.get("checkpoint") or (
        "best_val_model" if params.get("benchmark_member") else "best_model")


# ─── Feature extractors ──────────────────────────────────────────────────────
# CustomFeatureExtractor lives in extractors/euromlsys_extractor.py.
# Re-exported here so existing callers (transfer.py, etc.) keep working.

from extractors.euromlsys_extractor import CustomFeatureExtractor


class DictLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        embedding_size: int = 32,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        total_input_dim = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                input_dim = subspace.shape[0]
                self.extractors[key] = nn.Linear(input_dim, embedding_size)
                total_input_dim += embedding_size
            elif isinstance(subspace, spaces.MultiDiscrete):
                num_categories = subspace.nvec
                embedding_dims = [
                    min(embedding_size, (n // 2) + 1) for n in num_categories
                ]
                self.embeddings[key] = nn.ModuleList(
                    [
                        nn.Embedding(num_categories[i], embedding_dims[i])
                        for i in range(len(num_categories))
                    ]
                )
                total_input_dim += sum(embedding_dims)

        self.lstm = nn.LSTM(total_input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        extracted_features = []

        for key, extractor in self.extractors.items():
            if key in observations:
                extracted_features.append(extractor(observations[key]))

        for key, embedding_layers in self.embeddings.items():
            categorical_features = []
            for i, embedding in enumerate(embedding_layers):
                categorical_features.append(embedding(observations[key].long()[:, i]))
            extracted_features.append(torch.cat(categorical_features, dim=-1))

        concatenated = torch.cat(extracted_features, dim=-1)
        batch_size = next(iter(observations.values())).shape[0]
        concatenated = concatenated.view(batch_size, 1, -1)
        lstm_out, _ = self.lstm(concatenated)
        return self.fc(lstm_out[:, -1, :])


class BoxLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, 1, -1)
        lstm_out, _ = self.lstm(observations)
        return self.fc(lstm_out[:, -1, :])


class BoxLSTMPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=BoxLSTMFeatureExtractor
        )


class DictLSTMPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=DictLSTMFeatureExtractor
        )


class BoxLSTMMaskablePPOPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=BoxLSTMFeatureExtractor
        )


class DictLSTMMaskablePPOPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=DictLSTMFeatureExtractor
        )


# ─── Freezing helpers ────────────────────────────────────────────────────────

def get_host_count_from_train_dir(train_model_dir) -> int | None:
    match = re.search(r"(\d+)hosts", train_model_dir)
    if not match:
        match = re.search(r"(\d+)nodes", train_model_dir)
    if match:
        return int(match.group(1))
    return None


def _get_total_hosts(params):
    total_hosts = 0
    for datacenter in params.get("datacenters", []):
        for host_type in datacenter.get("hosts", []):
            total_hosts += host_type.get("amount", 0)
    return total_hosts


def compute_freeze_indices_for_multi_dc_obs(params, prev_host_count=None) -> dict:
    """
    Compute indices for freezing weights in multi-DC flat observation space.
    """
    max_hosts = params["max_hosts"]
    cur_host_count = _get_total_hosts(params)
    max_host_pes = params["max_host_pes"]
    infr_obs_upper_bound = max_host_pes + 1
    cur_start_idx = 3 * cur_host_count * infr_obs_upper_bound
    end_idx = 3 * max_hosts * infr_obs_upper_bound

    prev_start_idx = None
    if prev_host_count is not None:
        prev_start_idx = 3 * prev_host_count * infr_obs_upper_bound

    return {
        "cur_start_idx": cur_start_idx,
        "end_idx": end_idx,
        "prev_start_idx": prev_start_idx,
    }


def compute_freeze_indices_for_tree_obs(params, prev_host_count=None) -> dict:
    """
    Compute indices for freezing weights in tree-based observation space.
    """
    max_hosts = params["max_hosts"]
    host_count = _get_total_hosts(params)
    host_pes = params["datacenters"][0]["hosts"][0]["pes"]
    small_vm_pes = params["vm_types"][0]["pes"]
    min_job_pes = 1

    cur_max_vms = host_count * host_pes // small_vm_pes
    cur_max_jobs = host_count * host_pes // min_job_pes
    max_vms = max_hosts * host_pes // small_vm_pes
    max_jobs = max_hosts * host_pes // min_job_pes
    infr_obs_length = 1 + max_hosts + max_vms + max_jobs
    max_pes_per_node = max_hosts * host_pes

    cur_start_idx = (1 + host_count + cur_max_vms + cur_max_jobs) * (max_pes_per_node + 1)
    end_idx = infr_obs_length * (max_pes_per_node + 1)

    prev_start_idx = None
    if prev_host_count is not None:
        prev_max_vms = prev_host_count * host_pes // small_vm_pes
        prev_max_jobs = prev_host_count * host_pes // min_job_pes
        prev_start_idx = (1 + prev_host_count + prev_max_vms + prev_max_jobs) * (max_pes_per_node + 1)

    return {
        "cur_start_idx": cur_start_idx,
        "end_idx": end_idx,
        "prev_start_idx": prev_start_idx,
    }


def maybe_freeze_weights(model, params, prev_host_count=None) -> None:
    """
    Freeze or unfreeze input layer weights based on active/inactive regions.
    """
    if not params.get("freeze_inactive_input_layer_weights", False):
        return
    if params.get("state_space_type") == "tree" and params.get("vm_allocation_policy") == "rl":
        indices = compute_freeze_indices_for_tree_obs(params, prev_host_count)
    elif (
        params.get("state_space_type") == "dcid-dctype-freevmpes-per-host"
        and params.get("cloudlet_to_dc_mapping") == "rl"
    ):
        indices = compute_freeze_indices_for_multi_dc_obs(params, prev_host_count)
    else:
        return

    prev_start_idx = indices["prev_start_idx"]
    cur_start_idx = indices["cur_start_idx"]
    end_idx = indices["end_idx"]

    weights = model.policy.mlp_extractor.policy_net[0].weight
    with torch.no_grad():
        weights[:, cur_start_idx:end_idx].requires_grad = False
        if prev_start_idx is not None:
            weights[:, :cur_start_idx].requires_grad = True
            min_start_idx = min(cur_start_idx, prev_start_idx)
            weights[:, min_start_idx:end_idx] = 0
        else:
            weights[:, cur_start_idx:end_idx] = 0


# ─── gRPC worker management (from main) ──────────────────────────────────────

_worker_rank = None
_worker_params = None
_worker_jobs_json = None
_worker_base_port = None
_worker_env = None


def _init_grpc_subproc(rank, params, jobs_json, base_port):
    """
    Initializer for SubprocVecEnv workers using spawn.
    Each subprocess gets its rank passed via the initializer args,
    then creates its own gRPC connection and Java JVM.
    """
    global _worker_rank, _worker_params, _worker_jobs_json, _worker_base_port, _worker_env
    _worker_rank = rank
    _worker_params = params
    _worker_jobs_json = jobs_json
    _worker_base_port = base_port
    _worker_env = _create_grpc_env_for_rank(rank, params, jobs_json, base_port)


def _create_grpc_env_for_rank(rank, params, jobs_json, base_port=50051):
    """
    Create a VmManagementEnv in the current process, starting a Java JVM first.
    Must be called from within the subprocess after fork/spawn.
    """
    import subprocess as _subprocess
    import time as _time
    import socket as _socket

    port = base_port + rank

    jar_path = os.environ.get(
        "CLOUDSIM_GATEWAY_JAR",
        "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-" + _gateway_version() + ".jar",
    )
    experiment_id = os.environ.get("EXPERIMENT_ID", "default")
    sim_log_dir = os.environ.get("JAVA_SIM_LOG_DIR", "")
    log_dest = os.environ.get("JAVA_LOG_DESTINATION", "stdout")
    log_level = os.environ.get("JAVA_LOG_LEVEL", "INFO")
    save_experiment = os.environ.get("SAVE_EXPERIMENT", "true").lower()
    # -D properties MUST come before -jar, otherwise they go to program's args[] instead of JVM
    java_cmd = [
        "java",
        "-Dlog.level=" + log_level,
        "-Dlog.destination=" + log_dest,
        "-Dlog.saveExperiment=" + save_experiment,
        f"-Dexperiment.id={experiment_id}",
        "-jar", jar_path,
        "--grpc", str(port),
    ]
    if sim_log_dir:
        java_cmd.insert(5, f"-Dlog.simDir={sim_log_dir}")
    # When log_dest=file: stdout goes to DEVNULL, stderr goes to STDOUT (captured by Python)
    # When log_dest=stdout: both go to inherited (visible in docker logs)
    java_stdout = None if ("stdout" in log_dest) else _subprocess.DEVNULL
    proc = _subprocess.Popen(
        java_cmd,
        stdout=java_stdout,
        stderr=_subprocess.STDOUT,
        env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseG1GC -Xmx256m"},
    )

    deadline = _time.time() + 60
    while _time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Java gRPC server (port {port}) exited with code {proc.returncode}")
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            sock.settimeout(1)
            sock.connect(("localhost", port))
            sock.close()
            break
        except Exception:
            sock.close()
            _time.sleep(0.5)

    from gym_cloudsimplus.envs import VmManagementEnv, JobPlacementEnv
    from gym_cloudsimplus.cloud_sim_grpc_client import _detect_rl_problem
    # Use _detect_rl_problem to dispatch to domain-named envs
    rl_problem = _detect_rl_problem(params)
    if rl_problem == "job_placement":
        env = JobPlacementEnv(params={**params, "worker_rank": rank}, jobs_as_json=jobs_json,
                              host="localhost", port=port)
        if params.get("benchmark_member"):
            env.set_level_stream(*level_stream(params, rank))
    else:
        env = VmManagementEnv(params=params, jobs_as_json=jobs_json, host="localhost", port=port)
    env._java_proc = proc
    return env


def level_stream(params: dict, rank: int) -> tuple[LevelSource, LevelSampler]:
    """Where worker `rank` gets its per-episode instances: the RING-N member's levels, drawn
    from params["level_split"] (train: a random level from the worker's own stream seeded by
    (seed, rank); val/test/lockbox: the worker's share of the split, round-robin)."""
    horizon = params["max_episode_length"]
    if params["timestep_interval"] != 1.0 or horizon < levels.HORIZON:
        raise ValueError(f"RING-N levels need timestep_interval 1.0 and max_episode_length >= "
                         f"{levels.HORIZON} (arrivals end at t={levels.LAST_ARRIVAL}), got "
                         f"{params['timestep_interval']} and {horizon}")
    source = LevelSource(params["ring_manifest"], params["benchmark_member"],
                         [dc["name"] for dc in params["datacenters"]])
    sampler = LevelSampler(params["level_split"], rank, params.get("num_cpu", 1), params["seed"])
    return source, sampler


def _make_grpc_env_for_subproc():
    """Factory for SubprocVecEnv workers using spawn."""
    return _worker_env


def make_grpc_env(rank, params, jobs_json, num_cpu, base_port=50051, log_dir=None):
    """
    Legacy factory for DummyVecEnv workers using gRPC.
    Each subprocess spawns its own Java JVM running the CloudSim gRPC server.
    """
    import subprocess as _subprocess
    import time as _time
    import socket as _socket

    port = base_port + rank

    def _start_java():
        jar_path = os.environ.get(
            "CLOUDSIM_GATEWAY_JAR",
            "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-" + _gateway_version() + ".jar",
        )
        experiment_id = os.environ.get("EXPERIMENT_ID", "default")
        log_dest = os.environ.get("JAVA_LOG_DESTINATION", "stdout")
        log_level = os.environ.get("JAVA_LOG_LEVEL", "INFO")
        sim_log_dir = os.environ.get("JAVA_SIM_LOG_DIR", "")
        # -D properties MUST come before -jar, otherwise they go to program's args[] instead of JVM
        java_cmd = [
            "java",
            "-Dlog.level=" + log_level,
            "-Dlog.destination=" + log_dest,
            f"-Dexperiment.id={experiment_id}",
            "-jar", jar_path,
            "--grpc", str(port),
        ]
        if sim_log_dir:
            java_cmd.insert(5, f"-Dlog.simDir={sim_log_dir}")
        proc = _subprocess.Popen(
            java_cmd,
            stdout=None if ("stdout" in log_dest) else _subprocess.DEVNULL,
            stderr=_subprocess.STDOUT,
            env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseG1GC -Xmx256m"},
        )
        deadline = _time.time() + 60
        while _time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Java gRPC server (port {port}) exited with code {proc.returncode}")
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            try:
                sock.settimeout(1)
                sock.connect(("localhost", port))
                sock.close()
                break
            except Exception:
                sock.close()
                _time.sleep(0.5)
        return proc

    def _init():
        proc = _start_java()
        try:
            from gym_cloudsimplus.envs import VmManagementEnv
            env = VmManagementEnv(params=params, jobs_as_json=jobs_json, host="localhost", port=port)
        except Exception:
            proc.terminate()
            proc.wait()
            raise
        env._java_proc = proc
        return env

    return _init


def _make_grpc_factory(rank, params, jobs_json, base_port):
    """Create a factory function for a gRPC worker with specific rank."""
    def _factory():
        env = _create_grpc_env_for_rank(rank, params, jobs_json, base_port)
        return env
    return _factory


class ParallelBatchDummyVecEnv(DummyVecEnv):
    """DummyVecEnv that steps every worker concurrently.

    Each worker is a gRPC client of its own Java JVM, and a step spends nearly all its time
    waiting on that JVM with the GIL released, so running the workers' steps in threads
    overlaps the simulations. step_wait's per-worker body is DummyVecEnv's (SB3 2.4-2.8),
    and each worker only writes its own buffer slots.
    """

    def __init__(self, env_fns, num_envs=None):
        from concurrent.futures import ThreadPoolExecutor
        super().__init__(env_fns)
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def step_wait(self):
        list(self._executor.map(self._step_worker, range(self.num_envs)))
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                deepcopy(self.buf_infos))

    def _step_worker(self, env_idx):
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = \
            self.envs[env_idx].step(self.actions[env_idx])
        self.buf_dones[env_idx] = terminated or truncated
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
        if self.buf_dones[env_idx]:
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)

    def close(self):
        self._executor.shutdown(wait=False)
        super().close()


def vectorize_env(env, algorithm, num_cpu=None, params=None, jobs_json=None):
    """
    Wrap an environment for vectorized training.
    """
    from stable_baselines3.common.vec_env import VecMonitor

    num_cpu = num_cpu or min(16, os.cpu_count() or 8)
    base_port = params.get("grpc_base_port", 50051) if params else 50051

    # Pass log_dir to Java side so it writes csp.current.log alongside Python logs
    if params and params.get("log_dir"):
        os.environ["JAVA_SIM_LOG_DIR"] = params["log_dir"]

    env_fns = [_make_grpc_factory(i, params, jobs_json, base_port) for i in range(num_cpu)]
    env = ParallelBatchDummyVecEnv(env_fns, num_envs=num_cpu)

    class _VecMonitor(VecMonitor):
        def reset(self):
            obs = self.venv.reset()
            self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
            return obs

    env = _VecMonitor(env, params.get("log_dir", "/tmp/grpc_monitor") if params else "/tmp/grpc_monitor")
    return env


# ─── Algorithm & device ──────────────────────────────────────────────────────

def get_suitable_device(algorithm) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and algorithm != "A2C" else "cpu")


def _is_rl_mode(params) -> bool:
    """Domain-agnostic check: is RL the controlling policy?
    vm-management uses vm_allocation_policy; job-placement uses cloudlet_to_dc_mapping.
    """
    return (
        params.get("vm_allocation_policy") == "rl"
        or params.get("cloudlet_to_dc_mapping") == "rl"
    )


def check_reward_shaping_gamma(params: dict, model) -> None:
    """Potential-based shaping keeps the optimal policy only with the learner's own gamma;
    with any other gamma training still runs, but towards a different objective."""
    if params.get("reward_shaping") and params["reward_shaping_gamma"] != model.gamma:
        raise ValueError(
            f"reward_shaping_gamma={params['reward_shaping_gamma']} but "
            f"{type(model).__name__} uses gamma={model.gamma}"
        )


def get_algorithm(rl_algorithm_name, params) -> sb3.common.base_class.BaseAlgorithm:
    # Non-RL modes (rule-based / fromfile): pick PPO as a placeholder so the
    # env-creation pipeline still works. The model is never actually trained.
    if not _is_rl_mode(params):
        return getattr(sb3, "PPO")
    if hasattr(sb3, rl_algorithm_name):
        return getattr(sb3, rl_algorithm_name)
    if hasattr(sb3_contrib, rl_algorithm_name):
        return getattr(sb3_contrib, rl_algorithm_name)
    raise AttributeError(f"RL algorithm {rl_algorithm_name} not found.")


def maybe_load_replay_buffer(model, base_log_dir, train_model_dir) -> None:
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            base_log_dir, train_model_dir, "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)


def create_kwargs_with_algorithm_params(env, params) -> dict:
    algorithm_kwargs = {}
    if params.get("ent_coef") and params["rl_algorithm"] in ALGORITHMS_WITH_ENT_COEF:
        algorithm_kwargs["ent_coef"] = params["ent_coef"]
    if params.get("learning_rate") and params["rl_algorithm"] != "HER":
        algorithm_kwargs["learning_rate"] = params["learning_rate"]
    if params.get("n_rollout_steps") and params["rl_algorithm"] in ALGORITHMS_WITH_N_STEPS:
        algorithm_kwargs["n_steps"] = params["n_rollout_steps"]
    if params.get("batch_size") and params["rl_algorithm"] != "HER":
        algorithm_kwargs["batch_size"] = params["batch_size"]
    if params.get("seed") and params["rl_algorithm"] != "HER":
        algorithm_kwargs["seed"] = params["seed"]
    if params.get("action_noise") and params["rl_algorithm"] in ALGORITHMS_WITH_ACTION_NOISE:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=params["action_noise"] * np.ones(n_actions)
        )
        algorithm_kwargs["action_noise"] = action_noise
    if params.get("target_kl") and params["rl_algorithm"] == "PPO":
        algorithm_kwargs["target_kl"] = params["target_kl"]
    if params.get("n_epochs") and params["rl_algorithm"] in ("PPO", "MaskablePPO"):
        algorithm_kwargs["n_epochs"] = params["n_epochs"]
    return algorithm_kwargs


def create_policy_from_obs_space_type(observation_space):
    if isinstance(observation_space, gym.spaces.Dict) or isinstance(observation_space, gym.spaces.Tuple):
        return "MultiInputPolicy"
    return "MlpPolicy"


# ─── LSTM policy helpers (from euromlsys) ───────────────────────────────────

def create_correct_lstm_policy(observation_space, maskable) -> classmethod:
    if maskable:
        if isinstance(observation_space, spaces.Dict):
            return DictLSTMMaskablePPOPolicy
        return BoxLSTMMaskablePPOPolicy
    if isinstance(observation_space, spaces.Dict):
        return DictLSTMPPOPolicy
    return BoxLSTMPPOPolicy


def create_correct_policy(observation_space, params) -> str | classmethod:
    if params.get("use_lstm"):
        maskable = "Maskable" in params["rl_algorithm"]
        return create_correct_lstm_policy(observation_space, maskable)
    if isinstance(observation_space, gym.spaces.Dict) or isinstance(observation_space, gym.spaces.Tuple):
        return "MultiInputPolicy"
    return "MlpPolicy"
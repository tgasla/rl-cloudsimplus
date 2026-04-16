import re
import os
import yaml
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)
from utils.rl_algorithm_support_flags import (
    ALGORITHMS_WITH_ENT_COEF,
    ALGORITHMS_WITH_ACTION_NOISE,
    ALGORITHMS_WITH_N_STEPS,
)


def _gateway_version():
    """Read gatewayVersion from versions.gradle at runtime."""
    try:
        with open("/mgr/versions.gradle") as f:
            match = re.search(r"gatewayVersion\s*=\s*['\"]([^'\"]+)['\"]", f.read())
            return match.group(1) if match else "0.1.0"
    except FileNotFoundError:
        return "0.1.0"


def get_host_count_from_train_dir(train_model_dir):
    # Regular expression to match the number before "hosts" or "nodes", optionally delimited by an underscore
    match = re.search(r"(\d+)_?(hosts|nodes)", train_model_dir)
    if match:
        number = match.group(1)  # Extract the matched number
        return int(number)
    return None


def dict_from_config(replica_id, config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # first we read the common parameters and we overwrite them with the specific experiment parameters
    params = {**config["common"], **config[f"experiment_{replica_id}"]}
    return params


def create_logger(save_experiment, log_dir):
    log_destination = ["stdout"]
    if save_experiment:
        log_destination.extend(["csv", "tensorboard"])

    # the logger can write to stdout, progress.csv and tensorboard
    return configure(log_dir, log_destination)


def create_callback(save_experiment, log_dir, send_observation_tree_array=True):
    if save_experiment:
        return SaveOnBestTrainingRewardCallback(log_dir, send_observation_tree_array=send_observation_tree_array)
    # the callback writes all the .csv files and saves the model (with replay buffer) when the reward is the best
    return None


def compute_indices(params, prev_host_count=None):
    """
    Compute the indices and relevant parameters for freezing weights.

    Args:
        params (dict): Parameters containing host and VM configuration.
        prev_host_count (int, optional): Previous host count for transfer learning. Defaults to None.

    Returns:
        dict: A dictionary containing the computed indices and other derived values.
    """
    max_hosts = params["max_hosts"]
    host_count = params["host_count"]
    host_pes = params["host_pes"]
    small_vm_pes = params["small_vm_pes"]
    min_job_pes = 1

    cur_max_vms = host_count * host_pes // small_vm_pes
    cur_max_jobs = host_count * host_pes // min_job_pes
    max_vms = max_hosts * host_pes // small_vm_pes
    max_jobs = max_hosts * host_pes // min_job_pes
    infr_obs_length = 1 + max_hosts + max_vms + max_jobs
    max_pes_per_node = max_hosts * host_pes

    cur_start_idx = (1 + host_count + cur_max_vms + cur_max_jobs) * (
        max_pes_per_node + 1
    )
    end_idx = infr_obs_length * (max_pes_per_node + 1)

    prev_start_idx = None
    if prev_host_count is not None:
        prev_max_vms = prev_host_count * host_pes // small_vm_pes
        prev_max_jobs = prev_host_count * host_pes // min_job_pes
        prev_start_idx = (1 + prev_host_count + prev_max_vms + prev_max_jobs) * (
            max_pes_per_node + 1
        )

    return {
        "cur_start_idx": cur_start_idx,
        "end_idx": end_idx,
        "prev_start_idx": prev_start_idx,
    }


def maybe_freeze_weights(model, params, prev_host_count=None):
    """
    Freeze or unfreeze input layer weights based on active/inactive regions.

    Args:
        model (nn.Module): The model whose weights need to be adjusted.
        params (dict): Parameters containing host and VM configuration.
        prev_host_count (int, optional): Previous host count for transfer learning. Defaults to None.
    """
    indices = compute_indices(params, prev_host_count)
    prev_start_idx = indices["prev_start_idx"]
    cur_start_idx = indices["cur_start_idx"]
    end_idx = indices["end_idx"]

    # Access the weights of the input layer
    weights = model.policy.mlp_extractor.policy_net[0].weight

    with torch.no_grad():
        weights[:, cur_start_idx:end_idx].requires_grad = False

        if prev_start_idx is not None:
            weights[:, :cur_start_idx].requires_grad = True
            min_start_idx = min(cur_start_idx, prev_start_idx)
            weights[:, min_start_idx:end_idx] = 0
        else:
            weights[:, cur_start_idx:end_idx] = 0


# Global state for subprocess workers - set by _init_grpc_subproc
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

    # Start Java JVM and create GrpcSingleDC env in this subprocess
    _worker_env = _create_grpc_env_for_rank(rank, params, jobs_json, base_port)


def _create_grpc_env_for_rank(rank, params, jobs_json, base_port=50051):
    """
    Create a GrpcSingleDC env in the current process, starting a Java JVM first.
    Must be called from within the subprocess after fork/spawn.
    """
    import subprocess as _subprocess
    import time as _time
    import socket as _socket
    from grpc_cloudsimplus.envs.grpc_singledc import GrpcSingleDC

    port = base_port + rank

    # Start Java JVM
    # CLOUDSIM_GATEWAY_JAR env var overrides the default path.
    # The versioned filename must match shadowJar output; rebuild after bumping gatewayVersion.
    jar_path = os.environ.get(
        "CLOUDSIM_GATEWAY_JAR",
        "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-" + _gateway_version() + ".jar",
    )
    experiment_id = os.environ.get("EXPERIMENT_ID", "default")
    log_dest = os.environ.get("JAVA_LOG_DESTINATION", "stdout")
    java_cmd = [
        "java", "-jar", jar_path,
        "--grpc", str(port),
        "-Dlog.level=INFO",
        f"-Dexperiment.id={experiment_id}",
        f"-Dlog.destination={log_dest}",
    ]
    proc = _subprocess.Popen(
        java_cmd,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.STDOUT,
        env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC"},
    )

    # Wait for Java server to be ready
    deadline = _time.time() + 60
    while _time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Java gRPC server (port {port}) exited with code {proc.returncode}"
            )
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            sock.settimeout(1)
            sock.connect(("localhost", port))
            sock.close()
            break
        except Exception:
            sock.close()
            _time.sleep(0.5)

    # Create the GrpcSingleDC env - this connects via gRPC
    env = GrpcSingleDC(params=params, jobs_as_json=jobs_json, host="localhost", port=port)
    env._java_proc = proc
    return env


def _make_grpc_env_for_subproc():
    """
    Factory for SubprocVecEnv workers using spawn.
    Each subprocess already has _worker_env set by _init_grpc_subproc.
    """
    return _worker_env


def make_grpc_env(rank, params, jobs_json, num_cpu, base_port=50051, log_dir=None):
    """
    Legacy factory for DummyVecEnv workers using gRPC.
    Each subprocess spawns its own Java JVM running the CloudSim gRPC server.

    Args:
        rank:         Worker index (passed by SubprocVecEnv)
        params:       Simulation configuration dict
        jobs_json:    JSON string of job list
        num_cpu:      Total number of workers (used to compute port = base_port + rank)
        base_port:    Starting port for gRPC servers
        log_dir:      Directory for this worker's monitor.csv (default: /tmp/grpc_worker_{rank})
    """
    import subprocess as _subprocess
    import time as _time
    import socket as _socket
    from grpc_cloudsimplus.envs.grpc_singledc import GrpcSingleDC

    port = base_port + rank

    def _start_java():
        # Spawn a dedicated Java JVM for this worker
        jar_path = os.environ.get(
            "CLOUDSIM_GATEWAY_JAR",
            "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-" + _gateway_version() + ".jar",
        )
        experiment_id = os.environ.get("EXPERIMENT_ID", "default")
        log_dest = os.environ.get("JAVA_LOG_DESTINATION", "stdout")
        java_cmd = [
            "java", "-jar", jar_path,
            "--grpc", str(port),
            "-Dlog.level=INFO",
            f"-Dexperiment.id={experiment_id}",
            f"-Dlog.destination={log_dest}",
        ]
        proc = _subprocess.Popen(
            java_cmd,
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.STDOUT,
            env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC"},
        )

        # Wait for server to be ready (TCP socket check)
        deadline = _time.time() + 60
        import socket as _socket
        while _time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Java gRPC server (port {port}) exited with code {proc.returncode}"
                )
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
        # Start the Java JVM for this worker
        proc = _start_java()
        try:
            # Create the GrpcSingleDC - this connects via gRPC
            # No Monitor wrapper here - VecMonitor at DummyVecEnv level handles logging
            env = GrpcSingleDC(params=params, jobs_as_json=jobs_json, host="localhost", port=port)
        except Exception:
            proc.terminate()
            proc.wait()
            raise
        # Attach cleanup so env.close() also stops the JVM
        env._java_proc = proc
        return env

    return _init


def _make_grpc_factory(rank, params, jobs_json, base_port):
    """
    Create a factory function for a gRPC worker with specific rank.
    Each factory creates its own gRPC connection and Java JVM.
    """
    def _factory():
        env = _create_grpc_env_for_rank(rank, params, jobs_json, base_port)
        return env
    return _factory


class ParallelBatchDummyVecEnv:
    """
    Wraps a DummyVecEnv of GrpcSingleDC envs.

    Overrides step() to fire all 16 gRPC calls in parallel using a ThreadPoolExecutor,
    then return individual results as SB3 expects. This overlaps the 16 sequential
    roundtrips into a single parallel batch — ~16x fewer gRPC roundtrips in wall-clock time.

    Since each GrpcSingleDC env connects to its own Java JVM (separate process),
    the calls are independent and truly parallel on the Java side.
    """

    def __init__(self, env_fns, num_envs=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self._inner = DummyVecEnv(env_fns)
        self._num_envs = self._inner.num_envs
        self._executor = ThreadPoolExecutor(max_workers=self._num_envs)
        self._as_completed = as_completed
        # Cache method references for speed
        self._inner_reset = self._inner.reset
        self._inner_get_attr = self._inner.get_attr
        self._inner_method = self._inner.env_method
        self._inner_close = self._inner.close
        self._inner_seed = self._inner.seed
        self._inner_env_is_wrapped = getattr(self._inner, 'env_is_wrapped', lambda: False)

    @property
    def num_envs(self):
        return self._num_envs

    def reset(self, seed=None):
        # DummyVecEnv.reset() does not accept seed; seed is set via seed() instead
        return self._inner_reset()

    def step(self, actions):
        """
        Fire all env.step() calls in parallel, then return results one-by-one
        to match SB3's expected (obs, reward, done, info) format per env.
        """
        futures = []
        for i in range(self._num_envs):
            action_i = actions[i] if hasattr(actions, '__iter__') else actions
            futures.append(self._executor.submit(self._inner.envs[i].step, action_i))

        obss, rewards, dones, infos = [], [], [], []
        for f in futures:
            obs, reward, done, info = f.result()
            obss.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # DummyVecEnv stacks them into arrays
        return self._stack_results(obss, rewards, dones, infos)

    def _stack_results(self, obss, rewards, dones, infos):
        import numpy as np
        # Stack observations
        if isinstance(obss[0], dict):
            stacked = {}
            for key in obss[0]:
                vals = [o[key] for o in obss]
                stacked[key] = np.stack(vals)
            obs = stacked
        else:
            obs = np.stack(obss)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        return obs, rewards, dones, infos

    def get_attr(self, name, indices=None):
        return self._inner_get_attr(name, indices)

    def env_method(self, method_name, *method_args, **method_kwargs):
        return self._inner_method(method_name, *method_args, **method_kwargs)

    def close(self):
        self._executor.shutdown(wait=False)
        self._inner_close()

    def seed(self, seed=None):
        return self._inner_seed(seed)

    def env_is_wrapped(self, wrapper_class=None):
        return self._inner.env_is_wrapped(wrapper_class)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def vectorize_env(env, algorithm, num_cpu=None, params=None, jobs_json=None):
    """
    Wrap an environment for vectorized training.

    Args:
        env:        A single Gymnasium environment instance (can be None for gRPC mode)
        algorithm:  SB3 algorithm name (e.g., "PPO", "A2C")
        num_cpu:    Number of parallel workers. Defaults to min(16, os.cpu_count()).
        params:     Simulation params dict
        jobs_json:  JSON string of job list
    """
    from stable_baselines3.common.vec_env import VecMonitor

    num_cpu = num_cpu or min(16, os.cpu_count() or 8)
    base_port = params.get("grpc_base_port", 50051) if params else 50051

    # Use ParallelBatchDummyVecEnv: ThreadPoolExecutor fires all 16 gRPC calls
    # in parallel, overlapping the per-call roundtrip latency.
    env_fns = [_make_grpc_factory(i, params, jobs_json, base_port) for i in range(num_cpu)]
    env = ParallelBatchDummyVecEnv(env_fns, num_envs=num_cpu)
    env = VecMonitor(env, params.get("log_dir", "/tmp/grpc_monitor"))
    return env


def get_suitable_device(algorithm):
    return torch.device(
        "cuda" if torch.cuda.is_available() and algorithm != "A2C" else "cpu"
    )


def get_algorithm(algorithm_name, vm_allocation_policy):
    if vm_allocation_policy == "fromfile" or vm_allocation_policy == "rule-based":
        # If the vm_allocation_policy is fromfile or rule-based, pick a default algorithm
        # so the code triggers the simulation environment creation
        # NOTE: the algorithm decision through learning is not used at all in this case
        algorithm = getattr(sb3, "PPO")
    if vm_allocation_policy == "rl":
        if hasattr(sb3, algorithm_name):
            algorithm = getattr(sb3, algorithm_name)
        elif hasattr(sb3_contrib, algorithm_name):
            algorithm = getattr(sb3_contrib, algorithm_name)
        else:
            raise AttributeError(f"Algorithm {algorithm_name} not found.")
    else:
        raise AttributeError(f"Vm allocation policy {vm_allocation_policy} not found.")
    return algorithm


def maybe_load_replay_buffer(model, train_model_dir):
    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            train_model_dir,
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)


def create_kwargs_with_algorithm_params(env, params):
    algorithm_kwargs = {}
    if params.get("ent_coef") and params["algorithm"] in ALGORITHMS_WITH_ENT_COEF:
        algorithm_kwargs["ent_coef"] = params["ent_coef"]
    if params.get("learning_rate") and params["algorithm"] != "HER":
        algorithm_kwargs["learning_rate"] = params["learning_rate"]
    if params.get("n_rollout_steps") and params["algorithm"] in ALGORITHMS_WITH_N_STEPS:
        algorithm_kwargs["n_steps"] = params["n_rollout_steps"]
    if params.get("seed") and params["algorithm"] != "HER":
        algorithm_kwargs["seed"] = params["seed"]
    if (
        params.get("action_noise")
        and params["algorithm"] in ALGORITHMS_WITH_ACTION_NOISE
    ):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=params["action_noise"] * np.ones(n_actions)
        )
        algorithm_kwargs["action_noise"] = action_noise

    return algorithm_kwargs


def create_policy_from_obs_space_type(observation_space):
    if isinstance(observation_space, gym.spaces.Dict):
        return "MultiInputPolicy"  # when state is Spaces.Dict()
    return "MlpPolicy"  # when state is not Spaces.Dict()

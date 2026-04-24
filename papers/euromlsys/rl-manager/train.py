import os
import json
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from utils.misc import (
    create_logger,
    create_callback,
    maybe_freeze_weights,
    get_suitable_device,
    get_algorithm,
    create_kwargs_with_algorithm_params,
    create_correct_policy,
    _snake_to_camel,
    _params_to_java_format,
)

# Import GrpcMultiDC directly - no Py4J needed
from gym_cloudsimplus.envs.gym_multidc import MultiDC


def _make_grpc_env(rank, params, jobs_json, base_port=50051):
    """
    Create a GrpcMultiDC environment, spawning a Java JVM subprocess.
    Called once per worker in DummyVecEnv.
    """
    import subprocess
    import time
    import socket

    port = base_port + rank

    jar_path = os.environ.get(
        "CLOUDSIM_GATEWAY_JAR",
        "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-0.1.0.jar",
    )
    experiment_id = os.environ.get("EXPERIMENT_ID", "default")
    log_dest = os.environ.get("JAVA_LOG_DESTINATION", "stdout")
    java_cmd = [
        "java", "-jar", jar_path,
        "--grpc", str(port),
        "-Dlog.level=" + os.environ.get("JAVA_LOG_LEVEL", "INFO"),
        f"-Dexperiment.id={experiment_id}",
        f"-Dlog.destination={log_dest}",
    ]
    java_stdout = None if "stdout" in log_dest else subprocess.DEVNULL
    proc = subprocess.Popen(
        java_cmd,
        stdout=java_stdout,
        stderr=subprocess.STDOUT,
        env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC"},
    )

    # Wait for Java server to be ready
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Java gRPC server (port {port}) exited with code {proc.returncode}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(1)
            sock.connect(("localhost", port))
            sock.close()
            break
        except Exception:
            time.sleep(0.5)

    # Create the environment - this connects via gRPC
    env = GrpcMultiDC(params=params, jobs_as_json=jobs_json, host="localhost", port=port)
    env._java_proc = proc
    return env


def train(params, jobs):
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Convert jobs to JSON for gRPC
    jobs_json = json.dumps(jobs)

    # Create a single gRPC environment with its own Java JVM
    num_cpu = params.get("num_cpu", 1)
    base_port = params.get("grpc_base_port", 50051)

    # Convert params to Java camelCase format for gRPC
    java_params = _params_to_java_format(params)

    env_fns = [lambda rank=i: _make_grpc_env(rank, java_params, jobs_json, base_port) for i in range(num_cpu)]
    env = DummyVecEnv(env_fns)

    # Subclass VecMonitor to fix reset() — ppo_mask expects (obs,) but
    # VecMonitor.reset() returns (obs, info) tuple from the underlying vec env.
    class _VecMonitor(VecMonitor):
        def reset(self):
            obs = self.venv.reset()
            self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
            return obs

    env = _VecMonitor(env, params.get("log_dir", "/tmp/grpc_monitor"))

    device = get_suitable_device(params["algorithm"])

    algorithm_kwargs = create_kwargs_with_algorithm_params(env, params)

    policy = create_correct_policy(env.observation_space, params)

    policy_kwargs = None
    if params.get("feature_extractor") == "custom":
        from utils.misc import CustomFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=params["features_dim"],
                embedding_size=params["embedding_size"],
                hidden_dim=params["hidden_dim"],
                adaptation_bottleneck=params.get("adaptation_bottleneck", False),
            ),
        )

    # Instantiate the agent
    model = algorithm(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        device=device,
        **algorithm_kwargs,
    )
    maybe_freeze_weights(model, params)

    callback = create_callback(params["save_experiment"], params["log_dir"])
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()

    # Delete the model from memory
    del model
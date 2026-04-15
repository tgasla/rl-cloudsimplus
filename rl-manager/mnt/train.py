import os
import json
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
from stable_baselines3.common.monitor import Monitor

from utils.trace_utils import csv_to_cloudlet_descriptor
from utils.misc import (
    create_logger,
    create_callback,
    maybe_freeze_weights,
    vectorize_env,
    get_suitable_device,
    get_algorithm,
    create_kwargs_with_algorithm_params,
    create_policy_from_obs_space_type,
)
from grpc_cloudsimplus.envs.grpc_singledc import GrpcSingleDC


def train(params):
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Read jobs
    job_trace_path = os.path.join("mnt", "traces", f"{params['job_trace_filename']}")
    jobs = csv_to_cloudlet_descriptor(job_trace_path)
    jobs_json = json.dumps(jobs)

    # Determine whether to use gRPC vectorized training
    use_grpc = params.get("use_grpc", False)
    num_cpu = params.get("num_cpu", None)

    if use_grpc:
        # vectorize_env creates DummyVecEnv with workers that each spawn their own JVM
        # Each worker computes its own spaces from params (deterministic, no gRPC needed)
        env = vectorize_env(
            None,
            algorithm,
            use_grpc=True,
            num_cpu=num_cpu,
            params=params,
            jobs_json=jobs_json,
        )
    else:
        # Legacy Py4J mode
        env = gym.make("SingleDC-v0", params=params, jobs_as_json=jobs_json)
        env = Monitor(env, params["log_dir"])
        env = vectorize_env(env, algorithm)

    device = get_suitable_device(params["algorithm"])

    algorithm_kwargs = create_kwargs_with_algorithm_params(env, params)

    policy = create_policy_from_obs_space_type(env.observation_space)

    # Instantiate the agent
    model = algorithm(policy=policy, env=env, device=device, **algorithm_kwargs)

    maybe_freeze_weights(model, params)

    callback = create_callback(params["save_experiment"], params["log_dir"],
                             params.get("send_observation_tree_array", True))
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()

    # Delete the model from memory
    del model

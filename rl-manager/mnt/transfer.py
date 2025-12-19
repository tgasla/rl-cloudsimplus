import os
import gymnasium as gym
import importlib
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from utils.misc import (
    create_kwargs_with_algorithm_params,
    create_logger,
    maybe_create_callback,
    get_algorithm,
    maybe_freeze_weights,
    vectorize_env,
    get_suitable_device,
    maybe_load_replay_buffer,
    get_host_count_from_train_dir,
)

import gym_cloudsimplus

importlib.reload(gym_cloudsimplus)


def transfer(params, jobs):
    best_model_path = os.path.join(
        params["base_log_dir"],
        f"{params['train_model_dir']}",
        "best_model",
    )

    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs=jobs)
    env = Monitor(env, params["log_dir"])
    env = vectorize_env(env, algorithm)

    # Change any model parameters you want here
    custom_objects = create_kwargs_with_algorithm_params(env, params)

    device = get_suitable_device(params["algorithm"])

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        env=env,
        device=device,
        custom_objects=custom_objects,
    )

    # prev_host_count = get_host_count_from_train_dir(params["train_model_dir"])
    # maybe_freeze_weights(model, params, prev_host_count=prev_host_count)

    callback = maybe_create_callback(params["save_experiment"], params["log_dir"])
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    maybe_load_replay_buffer(model, params["train_model_dir"])

    # Retrain the agent initializing the weights from the saved agent
    # The right thing to do is to set reset_num_timesteps=True
    # This way, the learning restarts
    # The only problem is that tensorboard recognizes
    # it as a new model, but that's not a critical issue for now
    # see: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    model.learn(
        total_timesteps=params["timesteps"],
        reset_num_timesteps=True,
        log_interval=1,
        callback=callback,
    )

    env.close()
    del model

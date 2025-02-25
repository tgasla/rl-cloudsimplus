import gymnasium as gym
import torch
import gym_cloudsimplus  # noqa: F401
from stable_baselines3.common.monitor import Monitor

from utils.misc import (
    create_logger,
    create_callback,
    maybe_freeze_weights,
    vectorize_env,
    get_suitable_device,
    get_algorithm,
    create_kwargs_with_algorithm_params,
    create_correct_policy,
    CustomFeatureExtractor,
)


def train(params, jobs):
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs=jobs)
    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    #   add info_keywords if needed
    # If log_dir is None, it will not log anything
    env = Monitor(env, params.get("log_dir"))
    env = vectorize_env(env, params["algorithm"])

    device = get_suitable_device(params["algorithm"])

    algorithm_kwargs = create_kwargs_with_algorithm_params(env, params)

    policy = create_correct_policy(env.observation_space, params)

    policy_kwargs = None
    if params.get("feature_extractor"):
        features_extractor_kwargs = {}
        if params.get("features_dim"):
            features_extractor_kwargs["features_dim"] = params["features_dim"]
        if params.get("embedding_size"):
            features_extractor_kwargs["embedding_size"] = params["embedding_size"]
        if params.get("hidden_dims"):
            features_extractor_kwargs["hidden_dims"] = params["hidden_dims"]
        if params.get("feature_extractor_load_path"):
            features_extractor_kwargs["load_path"] = params[
                "feature_extractor_load_path"
            ]
        if params.get("embedding_size"):
            features_extractor_kwargs["embedding_size"] = params["embedding_size"]
        if params.get("dropout"):
            features_extractor_kwargs["dropout"] = params["dropout"]
        if params.get("activation_fn"):
            features_extractor_kwargs["activation_fn"] = torch.nn.__dict__[
                params["activation_fn"]
            ]()
        policy_kwargs = dict(
            features_extractor_class=globals()[params["feature_extractor"]],
            features_extractor_kwargs=features_extractor_kwargs,
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

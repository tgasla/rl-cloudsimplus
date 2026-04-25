import os
import json

from utils.misc import (
    create_logger,
    create_callback,
    maybe_freeze_weights,
    get_suitable_device,
    get_algorithm,
    create_kwargs_with_algorithm_params,
    create_correct_policy,
    vectorize_env,
)


def train(params, jobs):
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Convert jobs to JSON for gRPC
    jobs_json = json.dumps(jobs)

    # vectorize_env spawns gRPC workers with Java JVMs (ParallelBatchDummyVecEnv)
    num_cpu = params.get("num_cpu", 16)
    env = vectorize_env(
        None,  # env not needed for spawning - vectorize_env creates workers directly
        algorithm,
        num_cpu=num_cpu,
        params=params,
        jobs_json=jobs_json,
    )

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
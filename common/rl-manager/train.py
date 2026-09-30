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
    check_reward_shaping_gamma,
    create_val_callback,
)


def train(params, jobs):
    if params.get("benchmark_member") and params.get("level_split") != "train":
        raise ValueError("RING-N train runs play level_split train, got "
                         f"{params.get('level_split')!r}: val selects checkpoints, "
                         "test and lockbox are held out")
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["rl_algorithm"], params)

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

    device = get_suitable_device(params["rl_algorithm"])

    algorithm_kwargs = create_kwargs_with_algorithm_params(env, params)

    policy = create_correct_policy(env.observation_space, params)

    policy_kwargs = None
    feature_extractor_name = params.get("feature_extractor")
    if feature_extractor_name and feature_extractor_name != "default":
        from extractors import (build_extractor_kwargs, build_policy_head_kwargs,
                                get_extractor_class, get_policy_class)
        policy = get_policy_class(feature_extractor_name, policy)
        policy_kwargs = dict(
            features_extractor_class=get_extractor_class(feature_extractor_name),
            features_extractor_kwargs=build_extractor_kwargs(feature_extractor_name, params),
            **build_policy_head_kwargs(feature_extractor_name, params),
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
    check_reward_shaping_gamma(params, model)

    callback = create_callback(params["save_experiment"], params["log_dir"])
    val_env = None
    if params.get("benchmark_member") and params["save_experiment"]:
        callback, val_env = create_val_callback(params, num_cpu)
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()
    if val_env is not None:
        val_env.close()

    # Delete the model from memory
    del model
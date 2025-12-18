import gymnasium as gym
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
    if params.get("feature_extractor") == "custom":
        from extractors.custom_extractor import CustomFeatureExtractor

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=params["features_dim"],
                embedding_size=params["embedding_size"],
                hidden_dim=params["hidden_dim"],
                adaptation_bottleneck=params["adaptation_bottleneck"],
            ),
        )
    elif params.get("feature_extractor") == "turret":
        from extractors.turret_extractor import TurretGNNExtractor

        policy_kwargs = dict(
            features_extractor_class=TurretGNNExtractor,
            features_extractor_kwargs=dict(
                params=params,  # Pass full params so Extractor can read topology
                features_dim=params["features_dim"],
                hidden_dim=params["hidden_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                leaky_relu_negative_slope=params["leaky_relu_negative_slope"],
            ),
        )
    elif params.get("feature_extractor") == "attention":
        from extractors.attention_extractor import AttentionFeatureExtractor

        policy_kwargs = dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=params["features_dim"],
                hidden_dim=params["hidden_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                leaky_relu_negative_slope=params["leaky_relu_negative_slope"],
                use_masking=params["use_masking"],
                use_host_ids=params["use_host_ids"],
            ),
        )
    elif params.get("feature_extractor") == "attention_pooling":
        from extractors.attention_pooling_extractor import (
            AttentionPoolingFeatureExtractor,
        )

        policy_kwargs = dict(
            features_extractor_class=AttentionPoolingFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=params["features_dim"],
                hidden_dim=params["hidden_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                leaky_relu_negative_slope=params["leaky_relu_negative_slope"],
                use_masking=params["use_masking"],
                use_host_ids=params["use_host_ids"],
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
    # maybe_freeze_weights(model, params)

    callback = create_callback(params["save_experiment"], params["log_dir"])
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()

    # Delete the model from memory
    del model

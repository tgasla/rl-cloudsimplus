import os
import json

from utils.misc import (
    create_kwargs_with_algorithm_params,
    create_logger,
    create_callback,
    get_algorithm,
    maybe_freeze_weights,
    vectorize_env,
    get_suitable_device,
    maybe_load_replay_buffer,
    get_host_count_from_train_dir,
    check_reward_shaping_gamma,
    create_val_callback,
    source_checkpoint,
    apply_finetune_scope,
)


def transfer(params, jobs):
    if params.get("benchmark_member") and params.get("level_split") != "train":
        raise ValueError("RING-N transfer runs play level_split train, got "
                         f"{params.get('level_split')!r}: val selects checkpoints, "
                         "test and lockbox are held out")
    best_model_path = os.path.join(
        params["base_log_dir"],
        f"{params['train_model_dir']}",
        source_checkpoint(params),
    )

    algorithm = get_algorithm(params["rl_algorithm"], params)

    num_cpu = params.get("num_cpu", 16)
    jobs_json = json.dumps(jobs)

    # vectorize_env spawns gRPC workers with Java JVMs (ParallelBatchDummyVecEnv)
    env = vectorize_env(
        None,
        algorithm,
        num_cpu=num_cpu,
        params=params,
        jobs_json=jobs_json,
    )

    # Change any model parameters you want here
    custom_objects = create_kwargs_with_algorithm_params(env, params)

    device = get_suitable_device(params["rl_algorithm"])

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        env=env,
        device=device,
        custom_objects=custom_objects,
    )

    prev_host_count = get_host_count_from_train_dir(params["train_model_dir"])
    maybe_freeze_weights(model, params, prev_host_count=prev_host_count)
    apply_finetune_scope(model, params.get("finetune", "full"))
    check_reward_shaping_gamma(params, model)

    callback = create_callback(params["save_experiment"], params["log_dir"])
    val_env = None
    if params.get("benchmark_member") and params["save_experiment"]:
        callback, val_env = create_val_callback(params, num_cpu)
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    maybe_load_replay_buffer(model, params["base_log_dir"], params["train_model_dir"])

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
    if val_env is not None:
        val_env.close()
    del model
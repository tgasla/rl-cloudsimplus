import os
import json
import numpy as np
import pandas as pd

from utils.misc import (
    get_algorithm,
    get_suitable_device,
    maybe_load_replay_buffer,
    _create_grpc_env_for_rank,
    source_checkpoint,
)


def test(params, jobs):
    if params.get("benchmark_member"):
        raise ValueError("RING-N runs are evaluated with mode: evaluate, which plays every level "
                         "of level_split once")
    best_model_path = os.path.join(
        params["base_log_dir"],
        params["train_model_dir"],
        source_checkpoint(params),
    )

    algorithm = get_algorithm(params["rl_algorithm"], params)

    jobs_json = json.dumps(jobs)
    base_port = params.get("grpc_base_port", 50051)

    # Create a single gRPC environment for evaluation (num_cpu=1)
    env = _create_grpc_env_for_rank(0, params, jobs_json, base_port)

    device = get_suitable_device(params["rl_algorithm"])

    # Load the trained agent
    model = algorithm.load(best_model_path, env=env, device=device, seed=params["seed"])

    maybe_load_replay_buffer(model, params["base_log_dir"], params["train_model_dir"])

    n_eval_episodes = params.get("n_eval_episodes", 10)
    episodes_info = {"r": [], "l": []}

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        current_length = 0
        done = False
        while not done:
            current_length += 1
            action_masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        episodes_info["r"].append(episode_reward)
        episodes_info["l"].append(current_length)

    mean_r = float(np.mean(episodes_info["r"]))
    std_r = float(np.std(episodes_info["r"]))
    print(f"Evaluation ({n_eval_episodes} episodes): mean reward = {mean_r:.3f} ± {std_r:.3f}")

    if params["save_experiment"]:
        os.makedirs(params["log_dir"], exist_ok=True)
        progress_file = os.path.join(params["log_dir"], "evaluation.csv")
        episode_info_df = pd.DataFrame(episodes_info)
        episode_info_df.to_csv(progress_file, index=False)

    env.close()
    del model

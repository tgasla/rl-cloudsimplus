import os
import json
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
import pandas as pd

from sb3_contrib.common.maskable.utils import get_action_masks
from utils.trace_utils import csv_to_cloudlet_descriptor
from utils.misc import (
    get_algorithm,
    get_suitable_device,
    maybe_load_replay_buffer,
)


def test(params):
    jobs = csv_to_cloudlet_descriptor(
        os.path.join("traces", f"{params['job_trace_filename']}")
    )

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    best_model_path = os.path.join(
        params["base_log_dir"],
        params["train_model_dir"],
        "best_model",
    )

    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    device = get_suitable_device(params["algorithm"])

    # Load the trained agent
    model = algorithm.load(best_model_path, env=env, device=device, seed=params["seed"])

    maybe_load_replay_buffer(model, params["train_model_dir"])

    predict_kwargs = {}
    if params["algorithm"] == "MaskablePPO":
        predict_kwargs["get_action_masks"] = get_action_masks(env)

    episodes_info = {"r": [], "inv": [], "h_a": [], "v_u": [], "j_w": [], "l": []}
    current_step = 0

    obs, info = env.reset()
    # print(f"Environment reset. Obs: {obs} Info: {info}")
    episode_reward = 0
    ep_inv_rew = 0
    ep_host_alloc_cores_rew = 0
    ep_vm_unutil_cores_rew = 0
    ep_job_wait_rew = 0
    current_length = 0
    done = False
    while not done:
        current_length += 1
        current_step += 1
        action, _ = model.predict(obs, **predict_kwargs)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(
        # f"Step: {current_length}, obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}"
        # )
        episode_reward += reward
        ep_inv_rew += info["invalid_reward"]
        ep_host_alloc_cores_rew += info["running_vm_cores_reward"]
        ep_vm_unutil_cores_rew += info["unutilized_vm_cores_reward"]
        ep_job_wait_rew += info["job_wait_reward"]
        done = terminated or truncated

    # print(
    # f"Episode ended. Episode length: {current_length}, episode reward: {episode_reward}"
    # )

    episodes_info["r"].append(episode_reward)
    episodes_info["l"].append(current_length)
    episodes_info["inv"].append(ep_inv_rew)
    episodes_info["h_a"].append(ep_host_alloc_cores_rew)
    episodes_info["v_u"].append(ep_vm_unutil_cores_rew)
    episodes_info["j_w"].append(ep_job_wait_rew)

    if params["save_experiment"]:
        # Create folder if needed
        os.makedirs(params["log_dir"], exist_ok=True)
        progress_file = os.path.join(params["log_dir"], "evaluation.csv")
        episode_info_df = pd.DataFrame(episodes_info)
        episode_info_df.to_csv(progress_file, index=False)

    env.close()
    del model

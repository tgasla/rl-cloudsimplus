import os
import sys
from datetime import datetime
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import custom_agents
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)

from utils.filename_generator import generate_filename
from utils.argparser import parse_args
from utils.trace_utils import csv_to_cloudlet_descriptor


def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def main():
    learning_rate_dict = {
        "DQN": "0.00005",
        "DDPG": "0.0001",
        "A2C": "0.0002",
        "PPO": "0.0001",
    }

    monitor_info_keywords = (
        "ep_job_wait_rew_mean",
        "ep_util_rew_mean",
        "ep_valid_count",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    # if pretrain dir is blank then you should pretrain first,
    # otherwise load agent from pretrain dir

    experiment_id = generate_filename(
        algorithm_str=args.algorithm_str,
        pretrain_timesteps=args.pretrain_timesteps,
        pretrain_hosts=args.pretrain_hosts,
        pretrain_host_pes=args.pretrain_host_pes,
        pretrain_host_pe_mips=args.pretrain_host_pe_mips,
        pretrain_reward_job_wait_coef=args.pretrain_reward_job_wait_coef,
        pretrain_reward_util_coef=args.pretrain_reward_util_coef,
        pretrain_reward_invalid_coef=args.pretrain_reward_invalid_coef,
        pretrain_job_trace_filename=args.pretrain_job_trace_filename,
        pretrain_max_job_pes=args.pretrain_max_job_pes,
    )

    base_log_dir = "./logs/"

    # Select the appropriate algorithm
    if hasattr(sb3, args.algorithm_str):
        algorithm = getattr(sb3, args.algorithm_str)
        policy = "MlpPolicy"
    else:
        algorithm = getattr(custom_agents, args.algorithm_str)
        policy = "RngPolicy"

    if args.pretrain_dir == "":
        timestamp = datetime_to_str()
        filename_id = timestamp + "_" + experiment_id
        log_dir = os.path.join(base_log_dir, f"{filename_id}")
        # Read jobs
        jobs = csv_to_cloudlet_descriptor(
            f"mnt/traces/{args.pretrain_job_trace_filename}.csv"
        )

        # Create folder if needed
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the environment
        env = gym.make(
            "SingleDC-v0",
            max_timesteps_per_episode=args.max_timesteps_per_episode,
            datacenter_hosts_cnt=args.pretrain_hosts,
            host_pe_mips=args.pretrain_host_pe_mips,
            host_pes=args.pretrain_host_pes,
            reward_job_wait_coef=args.pretrain_reward_job_wait_coef,
            reward_util_coef=args.pretrain_reward_util_coef,
            reward_invalid_coef=args.pretrain_reward_invalid_coef,
            jobs_as_json=json.dumps(jobs),
            max_job_pes=args.pretrain_max_job_pes,
            simulation_speedup=args.simulation_speedup,
            render_mode="ansi",
            job_log_dir=log_dir,
        )

        # Monitor needs the environment to have a render_mode set
        # If render_mode is None, it will give a warning.
        menv = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

        # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
        if args.algorithm_str == "A2C":
            device = "cpu"
            venv = SubprocVecEnv([lambda: menv], start_method="fork")
        else:
            venv = DummyVecEnv([lambda: menv])

        # if hasattr(algorithm, "ent_coef"):
        # algorithm.ent_coef = 0.01
        # algorithm.learning_rate=0.1,
        # algorithm.clip_range=0.7,
        # algorithm.n_steps=1024

        # Instantiate the agent
        model = algorithm(
            policy=policy,
            env=venv,
            verbose=1,
            # tensorboard_log=base_log_dir,
            device=device,
            seed=np.random.randint(0, 2**32 - 1),
        )

        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        # TODO: A2C and PPO take a n_steps parameter also :) check it out

        # Add some action noise for exploration if applicable
        if hasattr(model, "action_noise"):
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
            model.action_noise = action_noise

        callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir)

        # Train the agent
        model.learn(
            total_timesteps=int(args.pretrain_timesteps),
            # tb_log_name=filename_id,
            log_interval=1,
            callback=callback,
        )

        # Close the environment and free the resources
        env.close()

        # Delete model
        del model

    # if no transfer timesteps is specified, exit
    if args.transfer_timesteps == "":
        exit()

    jobs = csv_to_cloudlet_descriptor(
        f"mnt/traces/{args.transfer_job_trace_filename}.csv"
    )

    new_experiment_id = generate_filename(
        algorithm_str=args.algorithm_str,
        transfer_timesteps=args.transfer_timesteps,
        transfer_hosts=args.transfer_hosts,
        transfer_host_pes=args.transfer_host_pes,
        transfer_host_pe_mips=args.transfer_host_pe_mips,
        transfer_job_trace_filename=args.transfer_job_trace_filename,
        transfer_max_job_pes=args.transfer_max_job_pes,
        transfer_reward_job_wait_coef=args.transfer_reward_job_wait_coef,
        transfer_reward_util_coef=args.transfer_reward_util_coef,
        transfer_reward_invalid_coef=args.transfer_reward_invalid_coef,
        pretrain_dir=args.pretrain_dir,
    )

    new_timestamp = datetime_to_str()
    new_filename_id = new_timestamp + f"_{new_experiment_id}"

    if args.pretrain_dir != "":
        log_dir = os.path.join(base_log_dir, f"{args.pretrain_dir}")

    new_log_dir = os.path.join(base_log_dir, f"{new_filename_id}")

    # Create folder if needed
    os.makedirs(new_log_dir, exist_ok=True)

    # Create and wrap the environment
    new_env = gym.make(
        "SingleDC-v0",
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        datacenter_hosts_cnt=args.transfer_hosts,
        host_pe_mips=args.transfer_host_pe_mips,
        host_pes=args.transfer_host_pes,
        reward_job_wait_coef=args.transfer_reward_job_wait_coef,
        reward_util_coef=args.transfer_reward_util_coef,
        reward_invalid_coef=args.transfer_reward_invalid_coef,
        jobs_as_json=json.dumps(jobs),
        max_job_pes=args.transfer_max_job_pes,
        simulation_speedup=args.simulation_speedup,
        render_mode="ansi",
        job_log_dir=new_log_dir,
    )

    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    new_env = Monitor(new_env, new_log_dir, info_keywords=monitor_info_keywords)

    best_model_path = os.path.join(log_dir, "best_model")

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        device=device,
        # tensorboard_log=base_log_dir,
        env=new_env,
        seed=np.random.randint(0, 2**32 - 1),
    )

    logger = configure(new_log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(log_dir, "best_model_replay_buffer")
        model.load_replay_buffer(best_replay_buffer_path)

    # Set the learning rate to a small initial value
    model.learning_rate = learning_rate_dict.get(args.algorithm_str)

    callback = SaveOnBestTrainingRewardCallback(
        log_dir=new_log_dir,
    )

    # Retrain the agent initializing the weights from the saved agent
    model.learn(
        total_timesteps=int(args.transfer_timesteps),
        # The right thing to do is to set reset_num_timesteps=True
        # This way, the learning restarts
        # The only problem is that tensorboard recognizes
        # it as a new model, but that's not a critical issue for now
        reset_num_timesteps=True,
        # tb_log_name=new_filename_id,
        log_interval=1,
        callback=callback,
    )

    env.close()


if __name__ == "__main__":
    main()

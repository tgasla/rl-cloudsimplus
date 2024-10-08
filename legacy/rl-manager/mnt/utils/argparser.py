import argparse
from dataclasses import dataclass


@dataclass
class Argumenmts:
    algorithm_str: str
    max_timesteps_per_episode: str
    pretrain_timesteps: str
    pretrain_hosts: str
    pretrain_host_pes: str
    pretrain_host_pe_mips: str
    pretrain_job_trace_filename: str
    pretrain_max_job_pes: str
    pretrain_reward_job_wait_coef: str
    pretrain_reward_util_coef: str
    pretrain_reward_invalid_coef: str
    pretrain_dir: str
    transfer_timesteps: str
    transfer_hosts: str
    transfer_host_pes: str
    transfer_host_pe_mips: str
    transfer_job_trace_filename: str
    transfer_max_job_pes: str
    transfer_reward_job_wait_coef: str
    transfer_reward_util_coef: str
    transfer_reward_invalid_coef: str


def parse_args() -> Argumenmts:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        choices=["DQN", "A2C", "PPO", "RNG", "DDPG", "HER", "SAC", "TD3"],
        help="The RL algorithm that is used for training",
    )
    parser.add_argument(
        "--max-timesteps-per-episode",
        type=str,
        help="The maximum number of timesteps allowed before resetting the episode",
    )
    parser.add_argument(
        "--pretrain-timesteps", type=str, help="The number of timesteps to pretrain"
    )
    parser.add_argument(
        "--pretrain-hosts", type=str, help="The number of hosts of the pretraining env"
    )
    parser.add_argument(
        "--pretrain-host-pes",
        type=str,
        help="The number of PEs of hosts of the pretraining env",
    )
    parser.add_argument(
        "--pretrain-host-pe-mips",
        type=str,
        help="The number of MIPS per PE for hosts of the pretraining env",
    )
    parser.add_argument(
        "--pretrain-job-trace-filename",
        type=str,
        help="The trace file for the pretraining",
    )
    parser.add_argument(
        "--pretrain-max-job-pes",
        type=str,
        help="The maximum amount of pes allowed for job waiting on pretraining",
    )
    parser.add_argument(
        "--pretrain-reward-job-wait-coef",
        type=str,
        help="The reward coefficient for the job waiting on pretraining",
    )
    parser.add_argument(
        "--pretrain-reward-util-coef",
        type=str,
        help="The reward coefficient for the utilization on pretraining",
    )
    parser.add_argument(
        "--pretrain-reward-invalid-coef",
        type=str,
        help="The reward coefficient for invalid actions on pretraining",
    )
    parser.add_argument("--pretrain-dir", type=str, help="The pratrain dir")
    parser.add_argument(
        "--transfer-timesteps",
        type=str,
        help="The number of timesteps to perform after the environment transfer",
    )
    parser.add_argument(
        "--transfer-hosts", type=str, help="The number of hosts of the transfer env"
    )
    parser.add_argument(
        "--transfer-host-pes",
        type=str,
        help="The number of PEs of hosts of the transfer env",
    )
    parser.add_argument(
        "--transfer-host-pe-mips",
        type=str,
        help="The number of MIPS per PE for hosts of the transfer env",
    )
    parser.add_argument(
        "--transfer-job-trace-filename",
        type=str,
        help="The trace file for the transfer env",
    )
    parser.add_argument(
        "--transfer-max-job-pes",
        type=str,
        help="The maximum amount of pes allowed for job waiting on transfer env",
    )
    parser.add_argument(
        "--transfer-reward-job-wait-coef",
        type=str,
        help="The reward coefficient for the job waiting on transfer env",
    )
    parser.add_argument(
        "--transfer-reward-util-coef",
        type=str,
        help="The reward coefficient for the utilization on transfer",
    )
    parser.add_argument(
        "--transfer-reward-invalid-coef",
        type=str,
        help="The reward coefficient for invalid actions on transfer",
    )

    args = parser.parse_args()

    algorithm_str = str(args.algo).upper()
    max_timesteps_per_episode = str(args.max_timesteps_per_episode)

    pretrain_timesteps = str(args.pretrain_timesteps)
    pretrain_hosts = str(args.pretrain_hosts)
    pretrain_host_pes = str(args.pretrain_host_pes)
    pretrain_host_pe_mips = str(args.pretrain_host_pe_mips)
    pretrain_job_trace_filename = str(args.pretrain_job_trace_filename)
    pretrain_max_job_pes = str(args.pretrain_max_job_pes)
    pretrain_reward_job_wait_coef = str(args.pretrain_reward_job_wait_coef)
    pretrain_reward_util_coef = str(args.pretrain_reward_util_coef)
    pretrain_reward_invalid_coef = str(args.pretrain_reward_invalid_coef)
    pretrain_dir = str(args.pretrain_dir)

    transfer_timesteps = str(args.transfer_timesteps)
    transfer_hosts = str(args.transfer_hosts)
    transfer_host_pes = str(args.transfer_host_pes)
    transfer_host_pe_mips = str(args.transfer_host_pe_mips)
    transfer_max_job_pes = str(args.transfer_max_job_pes)
    transfer_job_trace_filename = str(args.transfer_job_trace_filename)
    transfer_reward_job_wait_coef = str(args.transfer_reward_job_wait_coef)
    transfer_reward_util_coef = str(args.transfer_reward_util_coef)
    transfer_reward_invalid_coef = str(args.transfer_reward_invalid_coef)

    return Argumenmts(
        algorithm_str=algorithm_str,
        max_timesteps_per_episode=max_timesteps_per_episode,
        pretrain_timesteps=pretrain_timesteps,
        pretrain_hosts=pretrain_hosts,
        pretrain_host_pes=pretrain_host_pes,
        pretrain_host_pe_mips=pretrain_host_pe_mips,
        pretrain_job_trace_filename=pretrain_job_trace_filename,
        pretrain_max_job_pes=pretrain_max_job_pes,
        pretrain_reward_job_wait_coef=pretrain_reward_job_wait_coef,
        pretrain_reward_util_coef=pretrain_reward_util_coef,
        pretrain_reward_invalid_coef=pretrain_reward_invalid_coef,
        pretrain_dir=pretrain_dir,
        transfer_timesteps=transfer_timesteps,
        transfer_hosts=transfer_hosts,
        transfer_host_pes=transfer_host_pes,
        transfer_host_pe_mips=transfer_host_pe_mips,
        transfer_max_job_pes=transfer_max_job_pes,
        transfer_job_trace_filename=transfer_job_trace_filename,
        transfer_reward_job_wait_coef=transfer_reward_job_wait_coef,
        transfer_reward_util_coef=transfer_reward_util_coef,
        transfer_reward_invalid_coef=transfer_reward_invalid_coef,
    )

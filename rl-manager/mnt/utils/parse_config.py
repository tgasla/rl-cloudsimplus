import yaml


def dict_from_config(replica_id, config):
    # keys_to_extract = [
    #     "max_episode_length",
    #     "initial_s_vm_count",
    #     "initial_m_vm_count",
    #     "initial_l_vm_count",
    #     "timestep_interval",
    #     "split_large_jobs",
    #     "vm_hourly_cost",
    #     "host_count",
    #     "host_pes",
    #     "host_pe_mips",
    #     "host_ram",
    #     "host_storage",
    #     "host_bw",
    #     "small_vm_pes",
    #     "small_vm_ram" "small_vm_storage",
    #     "small_vm_bw",
    #     "reward_job_wait_coef",
    #     "reward_running_vm_cores_coef",
    #     "reward_unutilized_vm_cores_coef",
    #     "reward_invalid_coef",
    #     "max_job_pes",
    #     "vm_startup_delay",
    #     "vm_shutdown_delay",
    #     "medium_vm_multiplier",
    #     "large_vm_multiplier",
    #     "paying_for_the_full_hour",
    #     "clear_created_cloudlet_list",
    #     "algorithm_str",
    #     "timesteps",
    # ]

    with open(config, "r") as file:
        config = yaml.safe_load(file)

    params = {**config["common"], **config[f"exp_{replica_id}"]}

    # params = {key: config["common"][key] for key in keys_to_extract}
    # replica_params = {key: config[f"exp_{replica_id}"][key] for key in keys_to_extract}
    # params.update(replica_params)
    return params

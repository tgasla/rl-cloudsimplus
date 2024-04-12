def generate_filename(
    algorithm_str=None,
    pretrain_timesteps=None,
    pretrain_hosts=None,
    pretrain_host_pes=None,
    pretrain_host_pe_mips=None,
    pretrain_job_trace_filename=None,
    pretrain_max_job_pes=None,
    pretrain_reward_job_wait_coef=None,
    pretrain_reward_util_coef=None,
    pretrain_reward_invalid_coef=None,
    transfer_hosts=None,
    transfer_host_pes=None,
    transfer_host_pe_mips=None,
    transfer_job_trace_filename=None,
    transfer_max_job_pes=None,
    transfer_reward_job_wait_coef=None,
    transfer_reward_util_coef=None,
    transfer_reward_invalid_coef=None,
    transfer_timesteps=None,
    pretrain_dir=None
):
    if pretrain_dir is None:
        filename_id = (
            f"{algorithm_str}"
            f"_{_millify(pretrain_timesteps)}"
            f"_{pretrain_hosts}H"
            f"_{pretrain_host_pes}P"
            f"_{pretrain_host_pe_mips}M"
            f"_{pretrain_job_trace_filename}"
            f"_{pretrain_max_job_pes}MJC"
            f"_{pretrain_reward_job_wait_coef}Q"
            f"_{pretrain_reward_util_coef}U"
            f"_{pretrain_reward_invalid_coef}I"
        )
        if transfer_timesteps is not None:
            filename_id = (
                f"{filename_id}"
                f"_{_millify(transfer_timesteps)}"
                f"_{transfer_hosts}H"
                f"_{transfer_host_pes}P"
                f"_{transfer_host_pe_mips}M"
                f"_{transfer_job_trace_filename}"
                f"_{transfer_max_job_pes}MJC"
                f"_{transfer_reward_job_wait_coef}Q"
                f"_{transfer_reward_util_coef}U"
                f"_{transfer_reward_invalid_coef}I"
            )
    else:
         filename_id = (
            f"{pretrain_dir}"
            f"_{_millify(transfer_timesteps)}"
            f"_{transfer_hosts}H"
            f"_{transfer_host_pes}P"
            f"_{transfer_host_pe_mips}M"
            f"_{transfer_job_trace_filename}"
            f"_{transfer_max_job_pes}MJC"
            f"_{transfer_reward_job_wait_coef}Q"
            f"_{transfer_reward_util_coef}U"
            f"_{transfer_reward_invalid_coef}I"
        )
    return filename_id

def _millify(num):
    if isinstance(num, str):
        num = int(num)
        
    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ['', 'K', 'M', 'B', 'T']
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]
def generate_filename(
    algorithm_str,
    pretrain_timesteps,
    pretrain_hosts,
    pretrain_host_pes,
    pretrain_host_pe_mips,
    pretrain_job_trace_filename,
    pretrain_max_job_pes,
    pretrain_reward_job_wait_coef,
    pretrain_reward_vm_cost_coef,
    pretrain_reward_invalid_coef,
    transfer_hosts,
    transfer_host_pes,
    transfer_host_pe_mips,
    transfer_job_trace_filename,
    transfer_max_job_pes,
    transfer_reward_job_wait_coef,
    transfer_reward_vm_cost_coef,
    transfer_reward_invalid_coef,
    transfer_timesteps=None,
    pretrain_dir=None
):
    if pretrain_dir is None:
        filename_id = (
            f"{algorithm_str}"
            f"_{_millify(pretrain_timesteps)}"
            f"_H{pretrain_hosts}"
            f"_P{pretrain_host_pes}"
            f"_M{pretrain_host_pe_mips}"
            f"_{pretrain_job_trace_filename}"
            f"_MJC{pretrain_max_job_pes}"
            f"_Q{pretrain_reward_job_wait_coef}"
            f"_U{pretrain_reward_vm_cost_coef}"
            f"_I{pretrain_reward_invalid_coef}"
        )
        if transfer_timesteps is not None:

            filename_id = (
                f"{filename_id}"
                f"_{_millify(transfer_timesteps)}"
                f"_H{transfer_hosts}"
                f"_P{transfer_host_pes}"
                f"_M{transfer_host_pe_mips}"
                f"_{transfer_job_trace_filename}"
                f"_MJC{transfer_max_job_pes}"
                f"_Q{transfer_reward_job_wait_coef}"
                f"_U{transfer_reward_vm_cost_coef}"
                f"_I{transfer_reward_invalid_coef}"
            )
    else:
         filename_id = (
            f"{pretrain_dir}"
            f"_{_millify(transfer_timesteps)}"
            f"_H{transfer_hosts}"
            f"_P{transfer_host_pes}"
            f"_M{transfer_host_pe_mips}"
            f"_{transfer_job_trace_filename}"
            f"_MJC{transfer_max_job_pes}"
            f"_Q{transfer_reward_job_wait_coef}"
            f"_U{transfer_reward_vm_cost_coef}"
            f"_I{transfer_reward_invalid_coef}"
        )
    return filename_id

def _millify(num):
    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ['', 'K', 'M', 'B', 'T']
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]
from datetime import datetime


def generate_filename(
    algorithm_str=None,
    timesteps=None,
    hosts=None,
    host_pes=None,
    host_pe_mips=None,
    job_trace_filename=None,
    max_job_pes=None,
    reward_job_wait_coef=None,
    reward_running_vm_cores_coef=None,
    reward_unutilized_vm_cores_coef=None,
    reward_invalid_coef=None,
    train_model_dir=None,
    mode=None,
    hostname=None,
):
    if train_model_dir is None:
        filename_id = (
            f"{_datetime_to_str()}"
            f"_{algorithm_str}"
            f"_{_millify(timesteps)}"
            f"_{hosts}H"
            f"_{host_pes}P"
            f"_{host_pe_mips}M"
            f"_{job_trace_filename}"
            f"_{max_job_pes}MJC"
            f"_{reward_job_wait_coef}Q"
            f"_{reward_running_vm_cores_coef}R"
            f"_{reward_unutilized_vm_cores_coef}U"
            f"_{reward_invalid_coef}I"
            f"_{mode}"
            f"_{hostname}"
        )
    else:
        filename_id = (
            f"{train_model_dir}"
            f"_{_datetime_to_str()}"
            f"_{_millify(timesteps)}"
            f"_{hosts}H"
            f"_{host_pes}P"
            f"_{host_pe_mips}M"
            f"_{job_trace_filename}"
            f"_{max_job_pes}MJC"
            f"_{reward_job_wait_coef}Q"
            f"_{reward_running_vm_cores_coef}R"
            f"_{reward_unutilized_vm_cores_coef}U"
            f"_{reward_invalid_coef}I"
            f"_{mode}"
            f"_{hostname}"
        )
    return filename_id


def _datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def _millify(num):
    if isinstance(num, str):
        num = int(num)

    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ["", "K", "M", "B", "T"]
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]

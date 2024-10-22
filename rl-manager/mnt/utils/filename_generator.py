from datetime import datetime


def generate_filename(
    params,
    hostname=None,
):
    optional_params = {
        "timesteps": _millify(params.get("timesteps")),
        "host_count": f"{params.get("host_count")}H",
        "host_pes": f"{params.get("host_pes")}P",
        "host_pe_mips": f"{params.get("host_pe_mips")}M",
        "job_trace_filename": f"{params.get("job_trace_filename")}",
        "max_job_pes": f"{params.get("max_job_pes")}",
        "reward_job_wait_coef": f"{params.get("reward_job_wait_coef")}W",
        "reward_running_vm_cores_coef": f"{params.get("reward_running_vm_cores_coef")}R",
        "reward_unutilized_vm_cores_coef": f"{params.get("reward_unutilized_vm_cores_coef")}U",
        "reward_invalid_coef": f"{params.get("reward_invalid_coef")}I",
        "mode": params.get("mode"),
        "vm_allocation_policy": params.get("vm_allocation_policy"),
        "algorithm": params.get("algorithm"),
        "state_representation": params.get("state_representation"),
        "hostname": hostname,
    }

    filename_id = ""
    if params.get("train_model_dir"):
        filename_id += f"{params.get("train_model_dir")}_"

    filename_id += _datetime_to_str()

    filename_id += "".join(
        f"_{value}" for value in optional_params.values() if value is not None
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

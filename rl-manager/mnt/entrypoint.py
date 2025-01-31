import os
import json
import pycurl
import random
import shutil
import numpy as np
from io import BytesIO
import torch
import sys

import importlib
from utils.parse_config import dict_from_config
from utils.trace_utils import csv_to_cloudlet_descriptor

CONFIG_FILE = "config.yml"

sensitivity_mapping = {"tolerant": 0, "moderate": 1, "critical": 2}


def get_sensitivity_level(sensitivity_str: str) -> int:
    return sensitivity_mapping[sensitivity_str]


def _find_replica_id(hostname):
    response_buffer = BytesIO()
    # Define the socket path and container URL
    unix_socket_path = "/run/docker.sock"
    container_url = f"http://docker/containers/{hostname}/json"

    # Initialize a cURL object
    curl = pycurl.Curl()

    # Set cURL options
    curl.setopt(pycurl.UNIX_SOCKET_PATH, unix_socket_path)
    curl.setopt(pycurl.URL, container_url)
    curl.setopt(pycurl.WRITEFUNCTION, response_buffer.write)
    curl.perform()
    curl.close()

    response_data = response_buffer.getvalue().decode("utf-8")
    replica_id = json.loads(response_data)["Name"].split("-")[-1]
    return replica_id


def _get_max_job_pes(jobs):
    max_job_pes = max([job["cores"] for job in jobs])
    return max_job_pes


def _set_seed_for_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.cuda.synchronize()
    # torch.backends.cudnn.enabled = False


def _write_seed_to_file(seed, log_dir, filename="seed.txt"):
    """
    Creates a text file and writes the given seed number to it.

    Parameters:
        seed (int): The seed number to write.
        filename (str): The name of the file to create or overwrite. Default is 'seed.txt'.
    """
    filepath = os.path.join(log_dir, filename)
    try:
        with open(filepath, "w") as file:
            file.write(str(seed))
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def _translate_job_location_names_to_idx(jobs, datacenters):
    for job in jobs:
        location_idx = _get_dc_idx_by_name(job["location"], datacenters)
        job["location"] = location_idx
    return jobs


def _translate_sensitivity_str_to_levels(jobs):
    for job in jobs:
        job["delaySensitivity"] = get_sensitivity_level(job["delaySensitivity"])
    return jobs


def _translate_connect_to_names_to_idx(datacenters):
    for dc in datacenters:
        connect_to_idx = []
        for connection in dc["connect_to"]:
            idx = _get_dc_idx_by_name(connection, datacenters)
            connect_to_idx.append(idx)
        dc["connect_to"] = connect_to_idx
    return datacenters


def _check_if_datacenters_have_unique_names(datacenters):
    names = [dc["name"] for dc in datacenters]
    if len(names) != len(set(names)):
        raise ValueError("Datacenters must have unique names.")


def _get_dc_idx_by_name(name, datacenters):
    for idx, dc in enumerate(datacenters):
        if dc["name"] == name:
            return idx
    raise ValueError(f"Datacenter with name {name} not found in datacenters list.")


def main():
    num_experiments = int(os.getenv("NUM_EXPERIMENTS"))
    run_mode = os.getenv("RUN_MODE")
    if run_mode == "batch":
        hostname = os.getenv("HOSTNAME")
        experiment_id = _find_replica_id(hostname)
    elif run_mode == "serial":
        experiment_id = os.getenv("EXPERIMENT_ID")
    params = dict_from_config(experiment_id, CONFIG_FILE)

    job_trace_path = os.path.join("mnt", "traces", params["job_trace_filename"])
    jobs = csv_to_cloudlet_descriptor(job_trace_path)

    params.update(run_mode=run_mode)
    params.update(num_experiments=num_experiments)
    datacenters = [dc.to_dict() for dc in params["datacenters"]]
    _check_if_datacenters_have_unique_names(datacenters)
    datacenters = _translate_connect_to_names_to_idx(datacenters)
    params.update(datacenters=datacenters)
    params.update(max_job_pes=_get_max_job_pes(jobs))

    jobs = _translate_job_location_names_to_idx(jobs, params["datacenters"])
    jobs = _translate_sensitivity_str_to_levels(jobs)

    if params["seed"] == "random":
        params["seed"] = np.random.randint(0, sys.maxsize)
    else:
        _set_seed_for_all(params["seed"])

    params["log_dir"] = None
    if params["save_experiment"]:
        params["log_dir"] = os.path.join(
            params["base_log_dir"],
            params["experiment_dir"],
            params["experiment_name"],
        )
        # Create folder if needed
        os.makedirs(params["log_dir"], exist_ok=True)
        # Make a copy of the config file in the log directory
        shutil.copy(CONFIG_FILE, params["log_dir"])
        _write_seed_to_file(params["seed"], params["log_dir"])

    try:
        module = importlib.import_module(params["mode"])
    except ModuleNotFoundError:
        print(
            f"Mode {params['mode']} was not found. Available modes are: 'train', 'transfer', 'test'."
        )
    func = getattr(module, params["mode"])
    func(params, jobs)


if __name__ == "__main__":
    main()

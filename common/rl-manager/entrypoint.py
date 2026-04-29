import os
import random
import shutil
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import importlib
from utils.misc import dict_from_config, _check_datacenters_unique, _register_yaml_constructors
from utils.misc import _translate_connect_to_names_to_idx
from utils.misc import _translate_job_location_names_to_idx
from utils.misc import _translate_sensitivity_str_to_levels
from utils.trace_utils import csv_to_cloudlet_descriptor

CONFIG_FILE = "config.yml"


def set_seed_for_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)


def write_seed_to_file(seed, log_dir, filename="seed.txt"):
    filepath = os.path.join(log_dir, filename)
    try:
        with open(filepath, "w") as file:
            file.write(str(seed))
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def _load_raw_config():
    """Load raw config for globals (java_log_destination, java_log_level)."""
    import yaml
    _register_yaml_constructors()
    with open(CONFIG_FILE, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def main():
    num_experiments = int(os.getenv("NUM_EXPERIMENTS"))
    experiment_id = os.getenv("EXPERIMENT_ID")

    params = dict_from_config(experiment_id, CONFIG_FILE)

    # Load job trace once, pass to train/transfer/test
    job_trace_path = os.path.join("traces", params["job_trace_filename"])
    jobs = csv_to_cloudlet_descriptor(job_trace_path)

    params.update(num_experiments=num_experiments)

    # ── euromlsys job_placement path: preprocess datacenters and jobs ──
    if "datacenters" in params:
        datacenters = [dc.to_dict() for dc in params["datacenters"]]
        _check_datacenters_unique(datacenters)
        datacenters = _translate_connect_to_names_to_idx(datacenters)
        params["datacenters"] = datacenters
        jobs = _translate_job_location_names_to_idx(jobs, params["datacenters"])
        jobs = _translate_sensitivity_str_to_levels(jobs)

    # ── Seed ──
    if params.get("seed") == "random":
        params["seed"] = np.random.randint(0, sys.maxsize)
    else:
        set_seed_for_all(params["seed"])

    # ── Log dir ──
    params["log_dir"] = None
    if params.get("save_experiment"):
        params["log_dir"] = os.path.join(
            params["base_log_dir"],
            params["experiment_dir"],
            params["experiment_name"],
        )
        os.makedirs(params["log_dir"], exist_ok=True)
        shutil.copy(CONFIG_FILE, params["log_dir"])
        write_seed_to_file(params["seed"], params["log_dir"])

    # ── Propagate Java log settings to environment ──
    raw_config = _load_raw_config()
    globals_cfg = raw_config.get("globals", {})
    if "java_log_destination" in globals_cfg:
        os.environ["JAVA_LOG_DESTINATION"] = globals_cfg["java_log_destination"]
    if "java_log_level" in globals_cfg:
        os.environ["JAVA_LOG_LEVEL"] = globals_cfg["java_log_level"]

    # ── Dispatch to train/transfer/test ──
    try:
        module = importlib.import_module(params["mode"])
    except ModuleNotFoundError as e:
        print(f"ERROR: Mode '{params['mode']}' not found. Import error: {e}")
        print(f"sys.path = {sys.path[:3]}")
        print(f"Files in /mgr: {os.listdir('/mgr')}")
        raise

    func = getattr(module, params["mode"])
    func(params, jobs)


if __name__ == "__main__":
    main()
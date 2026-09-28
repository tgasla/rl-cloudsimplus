import os
import random
import shutil
import signal
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import importlib
from utils.misc import dict_from_config, _check_datacenters_unique, _register_yaml_constructors
from utils.misc import _check_datacenter_amounts_are_one
from utils.misc import _translate_connect_to_names_to_idx
from utils.misc import _translate_job_location_names_to_idx
from utils.misc import _translate_sensitivity_str_to_levels
from utils.trace_utils import csv_to_cloudlet_descriptor
from utils.run_dir import (
    RunDirExistsError,
    SKIP_EXIT_CODE,
    prepare_run_dir,
    write_run_status,
)

CONFIG_FILE = "config.yml"


def set_seed_for_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # only affects CNNs
        # torch.backends.cudnn.benchmark = False # only affects CNNs
        # torch.backends.cuda.enable_flash_sdp(False) # Slows down training
        # torch.backends.cuda.enable_mem_efficient_sdp(False) # Slows down training
        # torch.use_deterministic_algorithms(True, warn_only=False) # Slows down training
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    num_experiments = int(os.getenv("NUM_EXPERIMENTS"))
    experiment_id = int(os.getenv("EXPERIMENT_ID"))

    params = dict_from_config(experiment_id, CONFIG_FILE)

    # Load job trace once, pass to train/transfer/test
    job_trace_path = os.path.join("traces", params["job_trace_filename"])
    jobs = csv_to_cloudlet_descriptor(job_trace_path)

    params.update(num_experiments=num_experiments)

    # ── Domain → RL problem mapping ──────────────────────────────────────────
    domain = os.getenv("DOMAIN")
    if not domain:
        raise ValueError("DOMAIN env var is not set. Must be 'vm-management' or 'job-placement'.")
    if domain == "job-placement":
        params["rl_problem"] = "job_placement"
    elif domain == "vm-management":
        params["rl_problem"] = "vm_management"
    else:
        raise ValueError(f"DOMAIN must be 'vm-management' or 'job-placement', got: {domain}")

    # ── job-placement: preprocess custom datacenter objects and jobs ──
    if domain == "job-placement" and "datacenters" in params:
        datacenters = [dc.to_dict() for dc in params["datacenters"]]
        _check_datacenters_unique(datacenters)
        _check_datacenter_amounts_are_one(datacenters)
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
    save_experiment = params.get("save_experiment", False)
    params["log_dir"] = None
    if save_experiment:
        params["log_dir"] = prepare_run_dir(params)
        # Status first: a crash between here and the first result must still be
        # recognisable as an unfinished run, not mistaken for a legacy directory.
        write_run_status(params["log_dir"], params, "running")
        shutil.copy(CONFIG_FILE, params["log_dir"])

    os.environ["JAVA_LOG_DESTINATION"] = params.get("java_log_destination", "stdout")
    os.environ["JAVA_LOG_LEVEL"] = params.get("java_log_level", "INFO")
    os.environ["SAVE_EXPERIMENT"] = str(save_experiment).lower()

    # ── Dispatch to train/transfer/test ──
    try:
        module = importlib.import_module(params["mode"])
    except ModuleNotFoundError as e:
        print(f"ERROR: Mode '{params['mode']}' not found. Import error: {e}")
        print(f"sys.path = {sys.path[:3]}")
        print(f"Files in /mgr: {os.listdir('/mgr')}")
        raise

    func = getattr(module, params["mode"])

    status = "failed"
    try:
        func(params, jobs)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if params["log_dir"]:
            write_run_status(params["log_dir"], params, status)


def _raise_on_sigterm(signum, _frame):
    # startup.sh execs this module, so Python is PID 1 and would otherwise ignore
    # SIGTERM — `make stop-all` would leave the run recorded as still running.
    raise KeyboardInterrupt(f"terminated by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        main()
    except RunDirExistsError as exc:
        print(exc)
        sys.exit(SKIP_EXIT_CODE)
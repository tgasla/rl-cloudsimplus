import os
import json
import pycurl
import shutil
import numpy as np
from io import BytesIO

import importlib
from utils.parse_config import dict_from_config

CONFIG_FILE = "config.yml"
MAX_INT = 2**32 - 1


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


def main():
    hostname = os.getenv("HOSTNAME")
    replica_id = _find_replica_id(hostname)
    params = dict_from_config(replica_id, CONFIG_FILE)

    params["seed"] = (
        np.random.randint(0, MAX_INT) if params["seed"] == "random" else params["seed"]
    )

    params["log_dir"] = None
    if params["save_experiment"]:
        params["log_dir"] = os.path.join(
            params["base_log_dir"],
            params["experiment_type_dir"],
            params["experiment_name"],
        )
        # Create folder if needed
        os.makedirs(params["log_dir"], exist_ok=True)
        # Make a copy of the config file in the log directory
        shutil.copy(CONFIG_FILE, params["log_dir"])

    try:
        module = importlib.import_module(params["mode"])
        func = getattr(module, params["mode"])
        func(params)
    except ModuleNotFoundError:
        print(
            f"Mode {params['mode']} was not found. Available modes are: 'train', 'transfer', 'test'."
        )


if __name__ == "__main__":
    main()

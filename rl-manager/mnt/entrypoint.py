import os
import json
import pycurl
from io import BytesIO

import importlib
from utils.parse_config import dict_from_config


CONFIG_FILE = "config.yml"


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
    params["log_dir"] = os.path.join(
        params["base_log_dir"], params["experiment_type_dir"], params["experiment_dir"]
    )

    try:
        module = importlib.import_module(params["mode"])
        func = getattr(module, params["mode"])
        func(hostname, params)
    except ModuleNotFoundError:
        print(f"Module '{params['mode']}' could not be imported.")
    except AttributeError:
        print(
            f"The function '{params['mode']}' does not exist in the module '{params['mode']}'."
        )


if __name__ == "__main__":
    main()

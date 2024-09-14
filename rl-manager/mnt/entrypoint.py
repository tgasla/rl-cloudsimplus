import os
import json
import pycurl
from io import BytesIO
from utils.parse_config import dict_from_config
from train import train
from test import test
from transfer import transfer

CONFIG_FILE = "mnt/config.yml"


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

    mode = params["mode"]

    if mode == "train":
        train(hostname, params)
    elif mode == "transfer":
        transfer(hostname, params)
    elif mode == "test":
        test(hostname, params)
    else:
        ValueError("Unsupported mode. Supported modes are: [train, transter, test]")


if __name__ == "__main__":
    main()

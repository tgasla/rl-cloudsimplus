import yaml


def dict_from_config(replica_id, config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    params = {**config["common"], **config[f"exp_{replica_id}"]}
    return params

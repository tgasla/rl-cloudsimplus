import yaml


def dict_from_config(replica_id, config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    params = {**config["common"], **config[f"exp_{replica_id}"]}

    # params = {key: config["common"][key] for key in keys_to_extract}
    # replica_params = {key: config[f"exp_{replica_id}"][key] for key in keys_to_extract}
    # params.update(replica_params)
    return params

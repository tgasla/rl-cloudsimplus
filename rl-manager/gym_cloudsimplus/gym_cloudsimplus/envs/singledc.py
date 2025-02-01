import gymnasium as gym
import json
import os
import csv
import numpy as np
import torch
import re
from py4j.java_collections import JavaList, JavaArray
from gymnasium import spaces
from torch import nn
from torch.functional import F
from py4j.java_gateway import JavaGateway, GatewayParameters


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize the codebook (embedding vectors)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten input to (batch_size * num_features, embedding_dim)
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances between input and embedding vectors
        distances = torch.cdist(flat_x, self.embedding.weight, p=2)  # L2 distance
        indices = torch.argmin(distances, dim=1)  # Closest embedding index

        # Get quantized vectors
        quantized = self.embedding(indices).view(x.shape)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add quantization noise during backward pass
        quantized = x + (quantized - x).detach()
        return quantized, indices, loss


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_embeddings,
        dropout=0.0,
        use_batch_norm=False,
        commitment_cost=0.25,
    ):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # Assuming input values are normalized to [0, 1]
        )

    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Quantize
        quantized, indices, quantization_loss = self.quantizer(latent)
        # Decode
        reconstructed = self.decoder(quantized)
        return latent, quantized, reconstructed, quantization_loss


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),  # Assuming input values are normalized to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.0, use_batch_norm=False):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim) if use_batch_norm else nn.Identity(),
            nn.Sigmoid(),  # Output values are normalized to [0, 1]
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),  # Assuming input values are normalized to [0, 1]
        )

    def forward(self, x):
        # print("Input shape:", x.shape)
        latent = self.encoder(x)
        # print("Latent shape:", latent.shape)
        reconstructed = self.decoder(latent)
        # print("Reconstructed shape:", reconstructed.shape)
        return latent, reconstructed


# Based on https://gymnasium.farama.org/api/env/
class SingleDC(gym.Env):
    """
    Action space - Continuous

    A vector of 2 continuous numbers

    The first number is in range [-1,1]
    Positive value - create a VM
    The range shows the id of host that the VM will be placed

    Negative value - destroy a VM
    The range shows the id of the VM that will be destroyed

    The second number is in range [0,1]
    It shows the size of the VM that will be created/destroyed

    Action space - Discrete

    A vector of 3 discrete numbers ...

    Observation space
    """

    metadata = {"render_modes": ["human", "ansi"]}
    # default port = 25333
    gateway_parameters = GatewayParameters(address="gateway", auto_convert=True)
    params = {}

    def __init__(self, params, jobs, render_mode="ansi"):
        super(SingleDC, self).__init__()
        self.gateway = JavaGateway(gateway_parameters=self.gateway_parameters)
        self.simulation_environment = self.gateway.entry_point
        # self.reward_jobs_placed_coef = params.get("reward_jobs_placed_coef")
        # self.reward_quality_coef = params.get("reward_quality_coef")
        # self.reward_job_wait_coef = params.get("reward_job_wait_coef")
        # self.reward_running_vm_cores_coef = params.get("reward_running_vm_cores_coef")
        # self.reward_unutilized_vm_cores_coef = params.get(
        #     "reward_unutilized_vm_cores_coef"
        # )
        # self.reward_invalid_coef = params.get("reward_invalid_coef")
        # when read from the yaml file, params["datacenters"] is a list of Datacenter objects
        # we need to convert them to a list of dictionaries
        self.params = params

        self.autoencoder = None

        self.action_file_data = self._maybe_get_action_file_data()
        self.action_space = self._create_action_space()

        infr_obs_space, self.infr_obs_length = self._create_infr_obs_space()
        job_cores_waiting_obs_space, self.jobs_waiting_obs_length = (
            self._create_jobs_waiting_obs_space()
        )
        self.observation_space = self._create_obs_space(
            infr_obs_space, job_cores_waiting_obs_space
        )

        self.render_mode = self._check_render_mode(render_mode)
        params = self._convert_dict_keys_to_camel(params)
        params_as_json = json.dumps(params, default=self._json_serialization)
        jobs_as_json = json.dumps(jobs)
        self.simulation_id = self.simulation_environment.createSimulation(
            params_as_json, jobs_as_json
        )

    def _json_serialization(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return None

    def _snake_to_camel(self, snake_str):
        parts = snake_str.split("_")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])

    def _convert_dict_keys_to_camel(self, input_dict):
        return {self._snake_to_camel(key): value for key, value in input_dict.items()}

    def _camel_to_snake(self, camel_str):
        # Insert an underscore before each uppercase letter (except the first one) and convert to lowercase
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()

    def _convert_dict_keys_to_snake(self, input_dict):
        # Apply _camel_to_snake to each key in the dictionary
        return {self._camel_to_snake(key): value for key, value in input_dict.items()}

    def _get_datacenters_count(self):
        datacenter_count = 0
        for datacenter in self.params["datacenters"]:
            datacenter_count += datacenter["amount"]
        return datacenter_count

    def _check_render_mode(self, render_mode):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                'Invalid render modeRender modes allowed: ["human" | "ansi"]'
            )
            return None
        return render_mode

    def _get_infr_obs_length_from_type(self):
        obs_type = self.params["state_space_type"]
        if obs_type == "dcid-dctype-freevmpes-per-host":
            return 3 * self.params["max_hosts"]
        raise ValueError(f"Unknown observation type: {obs_type}")

    def _create_infr_obs_space(self):
        obs_type = self.params["state_space_type"]
        enable_autoencoder = self.params.get("enable_autoencoder_observation")
        latent_dim = self.params.get("autoencoder_latent_dim")
        if enable_autoencoder and latent_dim:
            input_dim = self._get_infr_obs_length_from_type()
            self.autoencoder = self._setup_autoencoder(
                input_dim, latent_dim, Autoencoder
            )
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(latent_dim,),
                dtype=np.float32,
            ), input_dim
        if obs_type == "dcid-dctype-freevmpes-per-host":
            infr_obs_length = (
                3 * self.params["max_hosts"]
            )  # dc id, free host pes (which in fact are free vm cores)
            infr_obs_upper_bound = self.params["max_host_pes"]
            return spaces.MultiDiscrete(  # + 1 to allow [0, max_host_pes] values
                (infr_obs_upper_bound + 1) * np.ones(infr_obs_length, dtype=np.int32)
            ), infr_obs_length
        if obs_type == "tree":
            # TODO: not ready yet, OUTDATED - DO NOT USE
            # Tree representation with MultiDiscrete
            infr_obs_length = 2 * (
                1 + self.total_hosts + self.total_vms + self.total_jobs
            )
            value_upper_bound = self._get_total_hosts() * self._get_max_host_pes()
            return spaces.MultiDiscrete(
                (value_upper_bound + 1) * np.ones(infr_obs_length, dtype=np.int32),
                dtype=np.int32,
            ), infr_obs_length
        if obs_type == "2d-array":
            # TODO: not ready yet
            # 2D array representation
            observation_rows = (
                self._get_datacenters_count()
                + self._get_total_hosts()
                + self._get_max_total_vms()
                + self._get_max_total_jobs()
            )
            observation_cols = 4  # 4 metrics for each entity
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(observation_rows, observation_cols),
                dtype=np.float32,
            ), observation_rows * observation_cols
        if obs_type == "autoencoder" and not latent_dim:
            raise ValueError(
                "autoencoder_latent_dim must be provided when using autoencoder observation type"
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

    def _create_jobs_waiting_obs_space(self):
        jobs_waiting_obs_length = 4 * self.params["max_jobs_waiting"]
        return spaces.MultiDiscrete(
            (self.params["max_job_pes"] + 1)
            * np.ones(jobs_waiting_obs_length, dtype=np.int32)
        ), jobs_waiting_obs_length

    def _create_obs_space(self, infr_obs_space, jobs_waiting_obs_space):
        return spaces.Dict(
            {
                "infr_state": infr_obs_space,
                "jobs_waiting_state": jobs_waiting_obs_space,
            }
        )

    def _create_action_space(self):
        return gym.spaces.MultiDiscrete(
            (self.params["max_datacenters"] + 1)
            * np.ones(self.params["max_jobs_waiting"], dtype=np.int32),
            # + 1 to allow for -1 which means the job queue is not full
            # datacenter ids in cloudsimplus start from 2 (LOL), so we start from 1 to allow a no action
        )  # the reason is that dcs are considered global entities and have global ids, CIS gets id 0, and broker gets id 1
        # return gym.spaces.Discrete(self._get_datacenters_count() + 1)

    def _setup_autoencoder(
        self,
        input_dim,
        latent_dim,
        ae_class=Autoencoder,
        ae_path=os.path.join("mnt", "autoencoders", "AE_5hosts_and_10hosts_64_BN.pth"),
    ) -> Autoencoder | SimpleAutoencoder | VQVAE | None:
        autoencoder = ae_class(input_dim, latent_dim, use_batch_norm=True)
        autoencoder.load_state_dict(torch.load(ae_path, weights_only=True))
        autoencoder.eval()
        return autoencoder

    # this is when infrastructure is [<dc_id, host_free_pes>, ...]
    def _get_max_cur_free_host_pes_per_dc(self):
        # Step 1: Create an empty dictionary to store the max free cores per datacenter
        max_cores_per_dc = {}

        # Step 2: Iterate over the list in pairs and update the dictionary with the maximum cores
        for i in range(0, len(self.infr_obs), 2):
            dc_id = self.infr_obs[i]  # Datacenter id
            free_cores = self.infr_obs[i + 1]  # Host free cores

            # Step 3: Check if the datacenter id is already in the dictionary
            if dc_id in max_cores_per_dc:
                # If it exists, update the max free cores if needed
                max_cores_per_dc[dc_id] = max(max_cores_per_dc[dc_id], free_cores)
            else:
                # If it does not exist, add it to the dictionary with the current free cores
                max_cores_per_dc[dc_id] = free_cores

        # Step 4: Convert the dictionary into a list of pairs (datacenter id, max free cores)
        result = [cores for cores in max_cores_per_dc.values()]
        # Print the result
        return result

    # calculated from the datacenter topology described in the yaml file
    # NOT from the parameter max_hosts_pes
    def _get_max_host_pes_per_dc(self):
        # Initialize the array with zeros (or use np.empty for uninitialized values)
        max_host_pes_per_dc = np.zeros(self._get_datacenters_count())
        # Iterate through datacenters and calculate max PEs per datacenter
        for i, datacenter in enumerate(self.params["datacenters"]):
            max_pes = max(host["pes"] for host in datacenter["hosts"])
            max_host_pes_per_dc[i] = max_pes  # Assign the max PEs to the array
        return max_host_pes_per_dc

    # calculated from the datacenter topology described in the yaml file
    # NOT from the parameter max_host_pes
    def _get_max_host_pes(self):
        return max(self._get_max_host_pes_per_dc())

    # calculated from the datacenter topology described in the yaml file
    # NOT from the parameter max_host_pes
    def _get_max_cur_host_pes(self):
        return max(self._get_max_cur_free_host_pes_per_dc())

    # calculated from the datacenter topology described in the yaml file
    def _get_connect_to_of_dc(self, dc_id):
        return self.params["datacenters"][int(dc_id)]["connect_to"]

    def action_masks(self) -> list[bool]:
        cur_dc_num = self._get_datacenters_count()  # Number of datacenters
        max_dc_num = self.params[
            "max_datacenters"
        ]  # Max number of datacenters (accounting for transfers)
        max_host_cores_per_dc = (
            self._get_max_cur_free_host_pes_per_dc()
        )  # Current max cores per datacenter
        # max_host_cores_all_dc = (
        #     self._get_max_cur_host_pes()
        # )  # Max cores across all datacenters
        # print("num_jobs", num_jobs)
        options_num = max_dc_num + 1  # Number of datacenters + 1 for no action
        # print("max_host_cores_all_dc", max_host_cores_all_dc)
        # print("job_cores_waiting_obs", self.jobs_waiting_obs)
        # Initialize valid_action_mask with False values
        # Each job can have dc_num + 1 actions (datacenters + no action)
        total_actions = self.params["max_jobs_waiting"] * (options_num)
        valid_action_mask = np.full(
            total_actions, False, dtype=bool
        )  # Start with all False

        for i in range(0, self.jobs_waiting_obs_length, 4):  # Iterate over jobs
            # Cores requested by the current job
            # each job has 4 values (cores, location, sensitivity, deadline)
            job_idx = i // 4
            job_cores = self.jobs_waiting_obs[i]
            job_location = self.jobs_waiting_obs[i + 1]
            # job_sensitivity = self.jpb_waiting_obs[i + 2]
            # job_deadline = self.jobs_waiting_obs[i + 3]
            # Iterate over datacenters (+1 for "no action" case)
            for j in range(options_num):
                # Compute the flat index for the action
                action_index = job_idx * options_num + j
                # If no cores it means no more jobs waiting
                # if j == 0 and job_cores == 0:
                #     # allow only the no action, to make the invalid action masking work
                #     valid_action_mask[action_index] = True
                # elif j == 0 and job_cores > max_host_cores_all_dc:
                #     # This case is when the job is too big for any host
                #     # So, allow only the no action, again
                #     valid_action_mask[action_index] = True
                if j == 0:
                    # allow no-op action for all jobs
                    valid_action_mask[action_index] = True
                elif (
                    0 < j <= cur_dc_num  # check if the datacenter id is valid
                    and (  # check connectivity of dcs
                        j - 1 in self._get_connect_to_of_dc(job_location)
                        or j - 1 == job_location
                    )
                    # check if there are enough cores in any host in the datacenter
                    and 0 < job_cores <= max_host_cores_per_dc[j - 1]
                ):
                    # because job cores must be > 0 it means that for jobs no waiting, the action is always 0,
                    # which means no-op.
                    # print(f"mpainw edw sto action index: {action_index}")
                    # print(f"job location: {job_location}")
                    # print(f"{self._get_connect_to_of_dc(job_location)}")
                    # Valid placement for the job
                    valid_action_mask[action_index] = True

        # if self.current_step == 30:
        #     print(self.jobs_waiting_obs)
        #     print("valid_action_mask: ", valid_action_mask.tolist())

        return valid_action_mask.tolist()  # Return as a Python list of booleans

    # def action_masks(self) -> list[bool]:
    #     """
    #     Return a list of masks for the current environment.
    #     The list should have the same length as the action space.
    #     Each element should be a list of booleans, where True means that the
    #     action is valid and False means that the action is invalid.
    #     """
    #     host_cores_utilized_sum = np.sum(self.host_cores_utilized)

    #     # the max vms the current datacenter can support (based on the number of hosts)
    #     current_max_vms = self.host_count * int(self.host_pes) // int(self.small_vm_pes)
    #     if host_cores_utilized_sum == 0:  # no VMs running
    #         action_type_mask = [True, True, False]
    #         host_mask = [True] * self.host_count + [False] * (
    #             self.max_hosts - self.host_count
    #         )
    #         vm_mask = (
    #             [True] + [False] * (self.max_vms - 1)
    #         )  # just allow one value to be True for the algorithm to work. It does not matter which one
    #         vm_type_mask = [True, True, False]
    #     elif self.vms_running == current_max_vms:  # all VMs are running
    #         action_type_mask = [True, False, True]
    #         host_mask = [True] + [False] * (self.max_hosts - 1)
    #         vm_mask = [True] * current_max_vms + False * (
    #             self.max_vms - current_max_vms
    #         )
    #         vm_type_mask = [True, False, False]
    #     elif all(
    #         self.host_pes - num < self.VM_CORES[0] for num in self.host_cores_utilized
    #     ):  # all hosts are full
    #         action_type_mask = [True, False, True]
    #         host_mask = [True] + [False] * (self.max_hosts - 1)
    #         vm_mask = [True] * self.vms_running + [False] * (
    #             self.max_vms - self.vms_running
    #         )
    #         vm_type_mask = [True, False, False]
    #     elif all(
    #         self.host_pes - num < self.VM_CORES[1] for num in self.host_cores_utilized
    #     ):  # can't create M, L VMs
    #         action_type_mask = [True, True, True]
    #         host_mask = [True] * self.host_count + [False] * (
    #             self.max_hosts - self.host_count
    #         )
    #         vm_mask = [True] * self.vms_running + [False] * (
    #             self.max_vms - self.vms_running
    #         )
    #         vm_type_mask = [True, False, False]
    #     elif all(
    #         self.host_pes - num < self.VM_CORES[2] for num in self.host_cores_utilized
    #     ):  # can't create L VMs
    #         action_type_mask = [True, True, True]
    #         host_mask = [True] * self.host_count + [False] * (
    #             self.max_hosts - self.host_count
    #         )
    #         vm_mask = [True] * self.vms_running + [False] * (
    #             self.max_vms - self.vms_running
    #         )
    #         vm_type_mask = [True, True, False]
    #     else:  # the most common case
    #         full_host_indices = [
    #             i
    #             for i, num in enumerate(self.host_cores_utilized)
    #             if self.host_pes - num < self.VM_CORES[0]
    #         ]
    #         action_type_mask = [True, True, True]
    #         host_mask = [
    #             False if i in full_host_indices else True
    #             for i in range(self.host_count)
    #         ] + [False] * (self.max_hosts - self.host_count)
    #         vm_mask = [True] * self.vms_running + [False] * (
    #             self.max_vms - self.vms_running
    #         )
    #         vm_type_mask = [True, True, True]

    #     masks = [action_type_mask, host_mask, vm_mask, vm_type_mask]
    #     return [item for sublist in masks for item in sublist]

    # Calculated from the datacenter topology described in the yaml file
    # NOT from the parameter max_hosts
    def _get_total_hosts(self):
        total_hosts = 0
        for datacenter in self.params["datacenters"]:
            for host_type in datacenter["hosts"]:
                total_hosts += host_type["amount"]
        return total_hosts

    def _maybe_get_action_file_data(self):
        if self.params["vm_allocation_policy"] == "fromfile":
            with open(
                os.path.join("mnt", "policies", self.params["algorithm"]), mode="r"
            ) as file:
                csv_reader = csv.reader(file)
                _ = next(csv_reader)  # skip the header
                self.action_file_data = list(csv_reader)

    def _pad_observation(self, obs, target_dim):
        obs_len = len(obs)
        if obs_len >= target_dim:
            return obs[:target_dim]  # Truncate if necessary
        # Create a zero array of target_dim and copy obs into it
        padded_obs = np.zeros(target_dim, dtype=obs.dtype)
        padded_obs[:obs_len] = obs
        return padded_obs

    def _min_max_normalize(self, array, min_val=0, max_val=100):
        """
        Normalize array to the range [0, 1] based on given min_val and max_val.
        """
        array = np.array(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)

    def _get_observation(self, result):
        raw_obs = result.getObservation()
        # print(type(raw_obs.getInfrastructureObservation()))
        infr_obs = self._convert_to_primitive(raw_obs.getInfrastructureObservation())

        # we need to pad the observation to the max length
        infr_obs = self._pad_observation(infr_obs, self.infr_obs_length)

        self.infr_obs = infr_obs

        infr_obs = self._maybe_get_autoencoder_obs(infr_obs)

        jobs_waiting_obs = self._convert_to_primitive(
            raw_obs.getJobsWaitingObservation()
        )
        jobs_waiting_obs = self._pad_observation(
            jobs_waiting_obs, self.jobs_waiting_obs_length
        )

        self.jobs_waiting_obs = jobs_waiting_obs

        return {
            "infr_state": infr_obs,
            "jobs_waiting_state": jobs_waiting_obs,
        }

    def _maybe_get_autoencoder_obs(self, infr_obs):
        if self.autoencoder:
            infr_obs = self._min_max_normalize(infr_obs)
            infr_obs = torch.tensor(infr_obs, dtype=torch.float32).unsqueeze(0)
            autoencoder_out = self.autoencoder(infr_obs)
            infr_obs = autoencoder_out[0].squeeze().detach().numpy()  # latent space
            return infr_obs
        return infr_obs

    def _get_action_from_file(self):
        if self.current_step - 1 >= len(self.action_file_data):
            raise ValueError(
                "The number of steps in the simulation exceeds the number of actions in the file"
            )
        action = self.action_file_data[self.current_step - 1]
        action = [int(x) for x in action]
        return action

    def _ensure_action_is_list(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, list):
            return action
        raise ValueError(f"Invalid action type: {type(action)}")

    def reset(self, seed=None, options=None):
        super(SingleDC, self).reset()
        self.current_step = 0
        # self.host_cores_utilized = np.zeros(self.max_hosts, dtype=np.int32)
        # self.vms_running = 0

        seed = 0 if seed is None else seed

        result = self.simulation_environment.reset(self.simulation_id, seed)

        obs = self._get_observation(result)
        self.jobs_waiting_obs = obs["jobs_waiting_state"]

        raw_info = result.getInfo()
        info = self._raw_info_to_dict(raw_info)
        return obs, info

    def step(self, action):
        # Py4J does not translate np.array(dtype=np.float32) to java List<double>
        # Fix1: make it dtype=np.float64 instead)
        # Fix2: before sending it to java, convert it to python list first
        # Here, we adopt Fix2
        self.current_step += 1

        if self.params["vm_allocation_policy"] == "fromfile":
            action = self._get_action_from_file()

        action = self._ensure_action_is_list(action)
        result = self.simulation_environment.step(self.simulation_id, action)

        reward = result.getReward()
        raw_info = result.getInfo()
        terminated = result.isTerminated()
        truncated = result.isTruncated()

        obs = self._get_observation(result)
        self.jobs_waiting_obs = obs["jobs_waiting_state"]

        # print(obs["infr_state"])

        ############################################################
        # ENABLE FOR ZERO OBSERVATION TESTING
        # obs = {
        #     "infr_state": np.zeros(self.infr_obs_length),
        #     "jobs_waiting_state": 0,
        # }
        ############################################################

        info = self._raw_info_to_dict(raw_info)

        if self.render_mode == "human":
            self.render()

        # Update the number of VMs per host
        # if info["isValid"]:
        #     # print(f"action: {action} was valid")
        #     if action[0] == 1:
        #         host_id = action[1]
        #         vm_type = action[3]
        #         self.host_cores_utilized[host_id] += self.VM_CORES[vm_type]
        #         self.vms_running += 1
        #     elif action[0] == 2:
        #         host_id = info["host_affected"]
        #         self.host_cores_utilized[host_id] -= info["cores_changed"]
        #         self.vms_running -= 1
        # else:
        #     print(f"action: {action} was invalid")

        # info = {}

        return (obs, reward, terminated, truncated, info)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method "
                "without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="human")'
            )
            return
        # result is a string with arrays encoded as json
        result = self.simulation_environment.render(self.simulation_id)
        obs_data = json.loads(result)
        obs_data_dict = dict(obs_data)
        if self.render_mode == "human":
            for key in obs_data:
                print(f"{key} -> {obs_data_dict[key]}")
            return
        elif self.render_mode == "ansi":
            return str(obs_data)
        else:
            return super(SingleDC, self).render()

    def close(self):
        # close the resources
        self.simulation_environment.close(self.simulation_id)
        # close the python client side
        self.gateway.close()

    def _raw_info_to_dict(self, raw_info):
        info = {}
        excluded_keys = ["class"]
        for attr in dir(raw_info):  # Get all attributes and methods of the object
            if attr.startswith("get") and callable(
                getattr(raw_info, attr)
            ):  # Look for "get" methods
                key = attr[3:]  # Remove the "get" prefix
                key = key[0].lower() + key[1:]  # Convert to camel case
                if key in excluded_keys:
                    continue
                try:
                    value = getattr(raw_info, attr)()  # Call the method
                    if (
                        isinstance(value, str)
                        and value.startswith("[")
                        and value.endswith("]")
                    ):
                        # Attempt to parse JSON strings
                        value = json.loads(value)
                    value = self._convert_to_primitive(value)
                    info[key] = value
                except Exception as e:
                    print(f"Error calling {attr}: {e}")
        info = self._convert_dict_keys_to_snake(info)
        return info

    def _convert_to_primitive(self, value):
        if isinstance(value, JavaList):
            return list(value)
        if isinstance(value, JavaArray):
            if len(value) == 0:
                return np.array(value)
            if isinstance(value[0], int):
                return np.array(value, dtype=np.int32)
            elif isinstance(value[0], float):
                return np.array(value, dtype=np.float32)
        return value

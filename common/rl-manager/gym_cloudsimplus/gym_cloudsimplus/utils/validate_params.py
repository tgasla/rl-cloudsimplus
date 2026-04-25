"""Configuration Validator — fail-fast incompatible param detection.

Fails immediately with a clear error message before any simulation starts.

Example:
    from utils.misc import validate_params
    validate_params(config)  # raises ConfigurationValidatorError if incompatible
"""

from ..cloud_sim_grpc_client import _detect_rl_problem


class ConfigurationValidatorError(ValueError):
    """Raised when configuration contains incompatible param combinations."""
    pass


VM_MANAGEMENT_PARAMS = {
    "host_count", "hosts_count", "host_pes", "host_pe_mips",
    "host_ram", "host_storage", "host_bw",
    "initial_s_vm_count", "initial_m_vm_count", "initial_l_vm_count",
    "small_vm_pes", "small_vm_ram", "small_vm_storage", "small_vm_bw",
    "small_vm_hourly_cost", "medium_vm_multiplier", "large_vm_multiplier",
    "reward_job_wait_coef", "reward_running_vm_cores_coef",
    "reward_unutilized_vm_cores_coef", "reward_invalid_coef",
}

JOB_PLACEMENT_PARAMS = {
    "datacenters", "max_datacenters", "max_datacenter_types",
    "cloudlet_to_dc_assignment_policy", "cloudlet_to_vm_assignment_policy",
    "state_space_type", "max_jobs_waiting", "job_trace_filename",
    "reward_jobs_placed_coef", "reward_quality_coef",
    "reward_deadline_violation_coef",
    "autoencoder_infr_obs", "autoencoder_infr_obs_latent_dim",
    "freeze_inactive_input_layer_weights",
}

INCOMPATIBLE_PAIRS = [
    ("vm_management", "job_arrival_rate", "job_arrival_rate cannot be used with VM_MANAGEMENT"),
    ("vm_management", "datacenters", "datacenters cannot be used with VM_MANAGEMENT"),
    ("vm_management", "max_datacenters", "max_datacenters cannot be used with VM_MANAGEMENT"),
    ("vm_management", "cloudlet_to_dc_assignment_policy", "cloudlet_to_dc_assignment_policy cannot be used with VM_MANAGEMENT"),
    ("job_placement", "host_count", "host_count cannot be used with JOB_PLACEMENT"),
    ("job_placement", "hosts_count", "hosts_count cannot be used with JOB_PLACEMENT"),
    ("job_placement", "small_vm_pes", "small_vm_pes cannot be used with JOB_PLACEMENT"),
    ("job_placement", "initial_s_vm_count", "initial_s_vm_count cannot be used with JOB_PLACEMENT"),
]


def validate_params(params: dict) -> str:
    """Validate param compatibility and return detected RL problem type.

    Raises ConfigurationValidatorError if incompatible params are found.
    Returns the RL problem type ('vm_management' or 'job_placement').
    """
    rl_problem = _detect_rl_problem(params)

    for rl_problem_key, param_key, message in INCOMPATIBLE_PAIRS:
        if rl_problem == rl_problem_key and param_key in params:
            raise ConfigurationValidatorError(
                f"Configuration error: {message} "
                f"(found {param_key}={params[param_key]}, but rl_problem={rl_problem})"
            )

    return rl_problem

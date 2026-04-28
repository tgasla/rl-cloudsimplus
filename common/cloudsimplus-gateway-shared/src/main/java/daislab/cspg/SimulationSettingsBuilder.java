package daislab.cspg;

import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 * Unified SimulationSettings builder for both RL problem types.
 *
 * Problem detection (in order):
 *   1. Explicit `rl_problem` key in params -> use directly
 *   2. `host_count` present            -> VM_MANAGEMENT
 *   3. `datacenters` / `max_datacenters` present -> JOB_PLACEMENT
 *
 * The builder pre-injects safe defaults for ALL fields of the INACTIVE problem
 * type BEFORE building. This prevents NPE when reading fields not present in
 * the current config -- inactive fields are simply 0/false/empty.
 */
public class SimulationSettingsBuilder {

    public static final String VM_MANAGEMENT = "vm_management";
    public static final String JOB_PLACEMENT = "job_placement";

    private SimulationSettingsBuilder() {} // static factory only

    /**
     * Build settings from params map.
     * Detects RL problem type and pre-injects defaults for inactive fields.
     * @return ISimulationSettings (SimulationSettingsVmManagement or SimulationSettingsJobPlacement)
     */
    public static ISimulationSettings build(Map<String, Object> params) {
        String rlProblem = detectRlProblem(params);
        return build(params, rlProblem);
    }

    /**
     * Build settings from params map with explicit RL problem type.
     * @return ISimulationSettings (SimulationSettingsVmManagement or SimulationSettingsJobPlacement)
     */
    public static ISimulationSettings build(Map<String, Object> params, String rlProblem) {
        // Pre-inject defaults for inactive problem type so inactive fields are never null
        Map<String, Object> filled = new HashMap<>(params);

        if (VM_MANAGEMENT.equals(rlProblem)) {
            injectJobPlacementDefaults(filled);
            return new SimulationSettingsVmManagement(filled);
        } else {
            injectVmManagementDefaults(filled);
            return new SimulationSettingsJobPlacement(filled);
        }
    }

    /**
     * Detect RL problem from params.
     * Priority: explicit key > host_count signal > datacenters signal.
     */
    public static String detectRlProblem(Map<String, Object> params) {
        // Explicit override
        Object explicit = params.get("rl_problem");
        if (explicit != null) {
            String s = explicit.toString();
            if (VM_MANAGEMENT.equalsIgnoreCase(s) || JOB_PLACEMENT.equalsIgnoreCase(s)) {
                return s.toLowerCase();
            }
        }

        // Signal: host_count means VM management (vm creation/destruction)
        if (params.containsKey("host_count") || params.containsKey("hosts_count")) {
            return VM_MANAGEMENT;
        }

        // Signal: datacenters means job placement
        if (params.containsKey("datacenters") || params.containsKey("max_datacenters")) {
            return JOB_PLACEMENT;
        }

        // Default: VM management (backward compat for existing main configs)
        return VM_MANAGEMENT;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Default injectors — called BEFORE the specific builder so inactive
    // fields get safe zero/empty values rather than NPE on access.
    // ─────────────────────────────────────────────────────────────────────────

    private static void injectVmManagementDefaults(Map<String, Object> filled) {
        // Job placement params that VM management doesn't have -> safe defaults
        filled.putIfAbsent("num_experiments", 1);
        filled.putIfAbsent("run_mode", "train");
        filled.putIfAbsent("reward_jobs_placed_coef", 0.333);
        filled.putIfAbsent("reward_quality_coef", 0.333);
        filled.putIfAbsent("reward_deadline_violation_coef", 0.333);
        filled.putIfAbsent("cloudlet_to_dc_assignment_policy", "rl");
        filled.putIfAbsent("cloudlet_to_vm_assignment_policy", "most-free-cores");
        filled.putIfAbsent("state_space_type", "dcid-dctype-freevmpes-per-host");
        filled.putIfAbsent("max_jobs_waiting", 50);
        filled.putIfAbsent("datacenters", List.of());
        filled.putIfAbsent("max_datacenters", 6);
        filled.putIfAbsent("max_datacenter_types", 3);
        filled.putIfAbsent("max_job_delay_sensitivity_levels", 3);
        filled.putIfAbsent("max_job_deadline", 20);
        filled.putIfAbsent("autoencoder_infr_obs", false);
        filled.putIfAbsent("autoencoder_infr_obs_latent_dim", 32);
        filled.putIfAbsent("freeze_inactive_input_layer_weights", false);
    }

    private static void injectJobPlacementDefaults(Map<String, Object> filled) {
        // VM management params that job placement doesn't have -> safe defaults
        filled.putIfAbsent("initial_s_vm_count", 0);
        filled.putIfAbsent("initial_m_vm_count", 0);
        filled.putIfAbsent("initial_l_vm_count", 0);
        filled.putIfAbsent("small_vm_hourly_cost", 0.0);
        filled.putIfAbsent("host_count", 0);
        filled.putIfAbsent("hosts_count", 0);
        filled.putIfAbsent("host_pe_mips", 0);
        filled.putIfAbsent("host_ram", 0);
        filled.putIfAbsent("host_storage", 0);
        filled.putIfAbsent("host_bw", 0);
        filled.putIfAbsent("small_vm_pes", 0);
        filled.putIfAbsent("small_vm_ram", 0);
        filled.putIfAbsent("small_vm_storage", 0);
        filled.putIfAbsent("small_vm_bw", 0);
        filled.putIfAbsent("medium_vm_multiplier", 0);
        filled.putIfAbsent("large_vm_multiplier", 0);
        filled.putIfAbsent("reward_job_wait_coef", 0.25);
        filled.putIfAbsent("reward_running_vm_cores_coef", 0.25);
        filled.putIfAbsent("reward_unutilized_vm_cores_coef", 0.25);
        filled.putIfAbsent("reward_invalid_coef", 0.25);
        filled.putIfAbsent("send_observation_tree_array", false);
        filled.putIfAbsent("algorithm", "PPO");
        filled.putIfAbsent("vm_allocation_policy", "rl");
    }
}

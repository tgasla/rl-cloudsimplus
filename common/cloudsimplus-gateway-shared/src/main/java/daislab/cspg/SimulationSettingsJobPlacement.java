package daislab.cspg;

import java.util.Map;
import java.util.List;
import lombok.Value;

/**
 * JOB_PLACEMENT simulation settings (euromlsys paper).
 * All 15 fields from the original euromlsys/SimulationSettings.java.
 * Safe defaults are pre-injected by SimulationSettingsBuilder for inactive fields.
 */
@Value
public class SimulationSettingsJobPlacement implements ISimulationSettings {
    // params map stored for reconstruction of paper-specific SimulationSettings
    Map<String, Object> params;

    String runMode;
    int numExperiments;
    double minTimeBetweenEvents;
    double timestepInterval;
    boolean splitLargeJobs;
    int maxJobPes;
    int maxHosts;
    double vmStartupDelay;
    double vmShutdownDelay;
    boolean payingForTheFullHour;
    boolean clearCreatedLists;
    double rewardJobsPlacedCoef;
    double rewardQualityCoef;
    double rewardDeadlineViolationCoef;
    int maxEpisodeLength;
    String vmAllocationPolicy;
    String cloudletToDcAssignmentPolicy;
    String cloudletToVmAssignmentPolicy;
    String algorithm;
    String stateSpaceType;
    int maxJobsWaiting;
    List<Map<String, Object>> datacenters;
    int maxDatacenters;
    int maxDatacenterTypes;
    int maxJobDelaySensitivityLevels;
    int maxJobDeadline;
    boolean autoencoderInfrObs;
    int autoencoderInfrObsLatentDim;
    boolean freezeInactiveInputLayerWeights;

    // Inactive VM management fields (pre-injected by builder)
    int initialSVmCount;
    int initialMVmCount;
    int initialLVmCount;
    int[] initialVmCounts;
    double smallVmHourlyCost;
    int hostsCount;
    long hostPeMips;
    int hostPes;
    int hostRam;
    int hostStorage;
    int hostBw;
    int smallVmPes;
    int smallVmRam;
    int smallVmStorage;
    int smallVmBw;
    int mediumVmMultiplier;
    int largeVmMultiplier;
    double rewardJobWaitCoef;
    double rewardRunningVmCoresCoef;
    double rewardUnutilizedVmCoresCoef;
    double rewardInvalidCoef;
    String algorithm_;
    boolean sendObservationTreeArray;

    public SimulationSettingsJobPlacement(Map<String, Object> params) {
        this.params = params;
        runMode = getStr(params, "run_mode", "train");
        numExperiments = getInt(params, "num_experiments", 1);
        minTimeBetweenEvents = getDouble(params, "min_time_between_events", 0.1);
        timestepInterval = getDouble(params, "timestep_interval", 1.0);
        splitLargeJobs = getBool(params, "split_large_jobs", false);
        maxJobPes = getInt(params, "max_job_pes", 16);
        maxHosts = getInt(params, "max_hosts", 40);
        vmStartupDelay = getDouble(params, "vm_startup_delay", 0.0);
        vmShutdownDelay = getDouble(params, "vm_shutdown_delay", 0.0);
        payingForTheFullHour = getBool(params, "paying_for_the_full_hour", false);
        clearCreatedLists = getBool(params, "clear_created_lists", false);
        rewardJobsPlacedCoef = getDouble(params, "reward_jobs_placed_coef", 0.333);
        rewardQualityCoef = getDouble(params, "reward_quality_coef", 0.333);
        rewardDeadlineViolationCoef = getDouble(params, "reward_deadline_violation_coef", 0.333);
        maxEpisodeLength = getInt(params, "max_episode_length", 150);
        vmAllocationPolicy = getStr(params, "vm_allocation_policy", "rl");
        cloudletToDcAssignmentPolicy = getStr(params, "cloudlet_to_dc_assignment_policy", "rl");
        cloudletToVmAssignmentPolicy = getStr(params, "cloudlet_to_vm_assignment_policy", "most-free-cores");
        algorithm = getStr(params, "algorithm", "PPO");
        stateSpaceType = getStr(params, "state_space_type", "dcid-dctype-freevmpes-per-host");
        maxJobsWaiting = getInt(params, "max_jobs_waiting", 50);
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
        maxDatacenters = getInt(params, "max_datacenters", 6);
        maxDatacenterTypes = getInt(params, "max_datacenter_types", 3);
        maxJobDelaySensitivityLevels = getInt(params, "max_job_delay_sensitivity_levels", 3);
        maxJobDeadline = getInt(params, "max_job_deadline", 20);
        autoencoderInfrObs = getBool(params, "autoencoder_infr_obs", false);
        autoencoderInfrObsLatentDim = getInt(params, "autoencoder_infr_obs_latent_dim", 32);
        freezeInactiveInputLayerWeights = getBool(params, "freeze_inactive_input_layer_weights", false);

        // Inactive VM management fields (pre-injected by builder)
        initialSVmCount = getInt(params, "initial_s_vm_count", 0);
        initialMVmCount = getInt(params, "initial_m_vm_count", 0);
        initialLVmCount = getInt(params, "initial_l_vm_count", 0);
        initialVmCounts = new int[] { initialSVmCount, initialMVmCount, initialLVmCount };
        smallVmHourlyCost = getDouble(params, "small_vm_hourly_cost", 0.0);
        hostsCount = getInt(params, "host_count", 0);
        hostPeMips = getInt(params, "host_pe_mips", 0);
        hostPes = getInt(params, "host_pes", 0);
        hostRam = getInt(params, "host_ram", 0);
        hostStorage = getInt(params, "host_storage", 0);
        hostBw = getInt(params, "host_bw", 0);
        smallVmPes = getInt(params, "small_vm_pes", 0);
        smallVmRam = getInt(params, "small_vm_ram", 0);
        smallVmStorage = getInt(params, "small_vm_storage", 0);
        smallVmBw = getInt(params, "small_vm_bw", 0);
        mediumVmMultiplier = getInt(params, "medium_vm_multiplier", 0);
        largeVmMultiplier = getInt(params, "large_vm_multiplier", 0);
        rewardJobWaitCoef = getDouble(params, "reward_job_wait_coef", 0.25);
        rewardRunningVmCoresCoef = getDouble(params, "reward_running_vm_cores_coef", 0.25);
        rewardUnutilizedVmCoresCoef = getDouble(params, "reward_unutilized_vm_cores_coef", 0.25);
        rewardInvalidCoef = getDouble(params, "reward_invalid_coef", 0.25);
        algorithm_ = getStr(params, "algorithm", "PPO");
        sendObservationTreeArray = getBool(params, "send_observation_tree_array", false);
    }

    // Safe accessors with defaults
    private static int getInt(Map<String, Object> m, String k, int def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).intValue();
        return Integer.parseInt(v.toString());
    }
    private static double getDouble(Map<String, Object> m, String k, double def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); } catch (Exception e) { return def; }
    }
    private static boolean getBool(Map<String, Object> m, String k, boolean def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(v.toString());
    }
    private static String getStr(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        return v == null ? def : v.toString();
    }
}

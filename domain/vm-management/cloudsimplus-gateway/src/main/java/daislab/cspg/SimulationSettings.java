package daislab.cspg;

import lombok.Value;
import java.util.Map;

/*
 * Class to describe the simulation settings. We provide two constructors.
 *
 * The first one that takes no parameters, creates the simulation by taking the settings from the
 * environment variables. If a parameter is not found as an environment variable, a default value is
 * given.
 *
 * The second one takes as a parameter a Map<String, String>. The first string represents the
 * parameter name and the second string represents the parameter value.
 */
@Value
public class SimulationSettings {
    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};

    double minTimeBetweenEvents;
    double timestepInterval;
    int initialSVmCount;
    int initialMVmCount;
    int initialLVmCount;
    int[] initialVmCounts;
    boolean splitLargeJobs;
    int maxJobPes;
    double smallVmHourlyCost;
    int maxHosts;
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
    double vmStartupDelay;
    double vmShutdownDelay;
    boolean payingForTheFullHour;
    boolean clearCreatedLists;
    double rewardJobWaitCoef;
    double rewardRunningVmCoresCoef;
    double rewardUnutilizedVmCoresCoef;
    double rewardInvalidCoef;
    int maxEpisodeLength;
    String vmAllocationPolicy;
    String algorithm;
    boolean sendObservationTreeArray;

    public SimulationSettings(final Map<String, Object> params) {
        minTimeBetweenEvents = getDouble(params, "min_time_between_events", 0.1);
        timestepInterval = getDouble(params, "timestep_interval", 1.0);
        initialSVmCount = getInt(params, "initial_s_vm_count", 0);
        initialMVmCount = getInt(params, "initial_m_vm_count", 0);
        initialLVmCount = getInt(params, "initial_l_vm_count", 0);
        initialVmCounts = new int[] {initialSVmCount, initialMVmCount, initialLVmCount};
        splitLargeJobs = getBool(params, "split_large_jobs", false);
        maxJobPes = getInt(params, "max_job_pes", 16);
        smallVmHourlyCost = getDouble(params, "small_vm_hourly_cost", 0.0);
        maxHosts = getInt(params, "max_hosts", 10);
        hostsCount = getInt(params, "host_count", getInt(params, "hosts_count", 0));
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
        vmStartupDelay = getDouble(params, "vm_startup_delay", 0.0);
        vmShutdownDelay = getDouble(params, "vm_shutdown_delay", 0.0);
        payingForTheFullHour = getBool(params, "paying_for_the_full_hour", false);
        clearCreatedLists = getBool(params, "clear_created_lists", false);
        rewardJobWaitCoef = getDouble(params, "reward_job_wait_coef", 0.25);
        rewardRunningVmCoresCoef = getDouble(params, "reward_running_vm_cores_coef", 0.25);
        rewardUnutilizedVmCoresCoef = getDouble(params, "reward_unutilized_vm_cores_coef", 0.25);
        rewardInvalidCoef = getDouble(params, "reward_invalid_coef", 0.25);
        maxEpisodeLength = getInt(params, "max_episode_length", 150);
        vmAllocationPolicy = getStr(params, "vm_allocation_policy", "rl");
        algorithm = getStr(params, "algorithm", "PPO");
        sendObservationTreeArray = getBool(params, "send_observation_tree_array", true);
    }

    private static int getInt(Map<String, Object> m, String k, int def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(v.toString()); } catch (Exception e) { return def; }
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

    // Lombok generates: all-args constructor, getters, equals, hashCode, toString

    public long getDatacenterCores() {
        return hostsCount * hostPes;
    }

    public long getTotalHostCores() {
        return hostsCount * hostPes;
    }

    public int getSizeMultiplier(final String type) {
        return switch (type) {
            case MEDIUM -> mediumVmMultiplier; // m5a.xlarge
            case LARGE -> largeVmMultiplier; // m5a.2xlarge
            case SMALL -> 1; // m5a.large
            default -> throw new IllegalArgumentException("Unexpected value: " + type);
        };
    }
}

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
public class SimulationSettings implements ISimulationSettings {
    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};

    Map<String, Object> params; // Lombok generates getParams()

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
        this.params = params;
        minTimeBetweenEvents = getDouble(params, "min_time_between_events");
        timestepInterval = getDouble(params, "timestep_interval");
        initialSVmCount = getInt(params, "initial_s_vm_count");
        initialMVmCount = getInt(params, "initial_m_vm_count");
        initialLVmCount = getInt(params, "initial_l_vm_count");
        initialVmCounts = new int[] {initialSVmCount, initialMVmCount, initialLVmCount};
        splitLargeJobs = getBool(params, "split_large_jobs");
        maxJobPes = getInt(params, "max_job_pes");
        smallVmHourlyCost = getDouble(params, "small_vm_hourly_cost");
        maxHosts = getInt(params, "max_hosts");
        hostsCount = getInt(params, "host_count");
        hostPeMips = getInt(params, "host_pe_mips");
        hostPes = getInt(params, "host_pes");
        hostRam = getInt(params, "host_ram");
        hostStorage = getInt(params, "host_storage");
        hostBw = getInt(params, "host_bw");
        smallVmPes = getInt(params, "small_vm_pes");
        smallVmRam = getInt(params, "small_vm_ram");
        smallVmStorage = getInt(params, "small_vm_storage");
        smallVmBw = getInt(params, "small_vm_bw");
        mediumVmMultiplier = getInt(params, "medium_vm_multiplier");
        largeVmMultiplier = getInt(params, "large_vm_multiplier");
        vmStartupDelay = getDouble(params, "vm_startup_delay");
        vmShutdownDelay = getDouble(params, "vm_shutdown_delay");
        payingForTheFullHour = getBool(params, "paying_for_the_full_hour");
        clearCreatedLists = getBool(params, "clear_created_lists");
        rewardJobWaitCoef = getDouble(params, "reward_job_wait_coef");
        rewardRunningVmCoresCoef = getDouble(params, "reward_running_vm_cores_coef");
        rewardUnutilizedVmCoresCoef = getDouble(params, "reward_unutilized_vm_cores_coef");
        rewardInvalidCoef = getDouble(params, "reward_invalid_coef");
        maxEpisodeLength = getInt(params, "max_episode_length");
        vmAllocationPolicy = getStr(params, "vm_allocation_policy");
        algorithm = getStr(params, "algorithm");
        sendObservationTreeArray = getBool(params, "send_observation_tree_array");
    }

    private static int getInt(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required int parameter: " + k);
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse int parameter '" + k + "': " + v);
        }
    }
    private static double getDouble(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required double parameter: " + k);
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse double parameter '" + k + "': " + v);
        }
    }
    private static boolean getBool(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required boolean parameter: " + k);
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(v.toString());
    }
    private static String getStr(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required string parameter: " + k);
        return v.toString();
    }

    public long getDatacenterCores() { return hostsCount * hostPes; }
    public long getTotalHostCores() { return hostsCount * hostPes; }
    public int getSizeMultiplier(final String type) {
        return switch (type) {
            case MEDIUM -> mediumVmMultiplier;
            case LARGE -> largeVmMultiplier;
            case SMALL -> 1;
            default -> throw new IllegalArgumentException("Unexpected value: " + type);
        };
    }
}
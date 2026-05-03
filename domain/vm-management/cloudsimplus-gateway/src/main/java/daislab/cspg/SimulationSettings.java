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
        minTimeBetweenEvents = ISimulationSettings.getDouble(params, "min_time_between_events");
        timestepInterval = ISimulationSettings.getDouble(params, "timestep_interval");
        initialSVmCount = ISimulationSettings.getInt(params, "initial_s_vm_count");
        initialMVmCount = ISimulationSettings.getInt(params, "initial_m_vm_count");
        initialLVmCount = ISimulationSettings.getInt(params, "initial_l_vm_count");
        initialVmCounts = new int[] {initialSVmCount, initialMVmCount, initialLVmCount};
        splitLargeJobs = ISimulationSettings.getBool(params, "split_large_jobs");
        maxJobPes = ISimulationSettings.getInt(params, "max_job_pes");
        smallVmHourlyCost = ISimulationSettings.getDouble(params, "small_vm_hourly_cost");
        maxHosts = ISimulationSettings.getInt(params, "max_hosts");
        hostsCount = ISimulationSettings.getInt(params, "host_count");
        hostPeMips = ISimulationSettings.getInt(params, "host_pe_mips");
        hostPes = ISimulationSettings.getInt(params, "host_pes");
        hostRam = ISimulationSettings.getInt(params, "host_ram");
        hostStorage = ISimulationSettings.getInt(params, "host_storage");
        hostBw = ISimulationSettings.getInt(params, "host_bw");
        smallVmPes = ISimulationSettings.getInt(params, "small_vm_pes");
        smallVmRam = ISimulationSettings.getInt(params, "small_vm_ram");
        smallVmStorage = ISimulationSettings.getInt(params, "small_vm_storage");
        smallVmBw = ISimulationSettings.getInt(params, "small_vm_bw");
        mediumVmMultiplier = ISimulationSettings.getInt(params, "medium_vm_multiplier");
        largeVmMultiplier = ISimulationSettings.getInt(params, "large_vm_multiplier");
        vmStartupDelay = ISimulationSettings.getDouble(params, "vm_startup_delay");
        vmShutdownDelay = ISimulationSettings.getDouble(params, "vm_shutdown_delay");
        payingForTheFullHour = ISimulationSettings.getBool(params, "paying_for_the_full_hour");
        clearCreatedLists = ISimulationSettings.getBool(params, "clear_created_lists");
        rewardJobWaitCoef = ISimulationSettings.getDouble(params, "reward_job_wait_coef");
        rewardRunningVmCoresCoef = ISimulationSettings.getDouble(params, "reward_running_vm_cores_coef");
        rewardUnutilizedVmCoresCoef = ISimulationSettings.getDouble(params, "reward_unutilized_vm_cores_coef");
        rewardInvalidCoef = ISimulationSettings.getDouble(params, "reward_invalid_coef");
        maxEpisodeLength = ISimulationSettings.getInt(params, "max_episode_length");
        vmAllocationPolicy = ISimulationSettings.getStr(params, "vm_allocation_policy");
        algorithm = ISimulationSettings.getStr(params, "algorithm");
        sendObservationTreeArray = ISimulationSettings.getBool(params, "send_observation_tree_array");
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
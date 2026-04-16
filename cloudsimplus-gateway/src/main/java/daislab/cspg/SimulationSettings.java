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
        minTimeBetweenEvents = 0.1;
        timestepInterval = (double) params.get("timestep_interval");
        initialSVmCount = (int) params.get("initial_s_vm_count");
        initialMVmCount = (int) params.get("initial_m_vm_count");
        initialLVmCount = (int) params.get("initial_l_vm_count");
        initialVmCounts = new int[] {initialSVmCount, initialMVmCount, initialLVmCount};
        splitLargeJobs = (boolean) params.get("split_large_jobs");
        maxJobPes = (int) params.get("max_job_pes");
        smallVmHourlyCost = (double) params.get("small_vm_hourly_cost");
        maxHosts = (int) params.get("max_hosts");
        hostsCount = (int) params.get("host_count");
        hostPeMips = (int) params.get("host_pe_mips");
        hostPes = (int) params.get("host_pes");
        hostRam = (int) params.get("host_ram");
        hostStorage = (int) params.get("host_storage");
        hostBw = (int) params.get("host_bw");
        smallVmPes = (int) params.get("small_vm_pes");
        smallVmRam = (int) params.get("small_vm_ram");
        smallVmStorage = (int) params.get("small_vm_storage");
        smallVmBw = (int) params.get("small_vm_bw");
        mediumVmMultiplier = (int) params.get("medium_vm_multiplier");
        largeVmMultiplier = (int) params.get("large_vm_multiplier");
        vmStartupDelay = (double) params.get("vm_startup_delay");
        vmShutdownDelay = (double) params.get("vm_shutdown_delay");
        payingForTheFullHour = (boolean) params.get("paying_for_the_full_hour");
        clearCreatedLists = (boolean) params.get("clear_created_lists");
        rewardJobWaitCoef = (double) params.get("reward_job_wait_coef");
        rewardRunningVmCoresCoef = (double) params.get("reward_running_vm_cores_coef");
        rewardUnutilizedVmCoresCoef = (double) params.get("reward_unutilized_vm_cores_coef");
        rewardInvalidCoef = (double) params.get("reward_invalid_coef");
        maxEpisodeLength = (int) params.get("max_episode_length");
        vmAllocationPolicy = (String) params.get("vm_allocation_policy");
        algorithm = (String) params.get("algorithm");
        sendObservationTreeArray = params.containsKey("send_observation_tree_array")
                ? (boolean) params.get("send_observation_tree_array")
                : true;
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

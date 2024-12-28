package daislab.cspg;

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
public class SimulationSettings {
    public final String SMALL = "S";
    public final String MEDIUM = "M";
    public final String LARGE = "L";
    public final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};
    private final double minTimeBetweenEvents = 0.1;
    private final double timestepInterval;
    private final int initialSVmCount;
    private final int initialMVmCount;
    private final int initialLVmCount;
    private final int[] initialVmCounts;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final double smallVmHourlyCost;
    private final int maxHosts;
    private final int hostsCount;
    private final long hostPeMips;
    private final int hostPes;
    private final int hostRam;
    private final int hostStorage;
    private final int hostBw;
    private final int smallVmPes;
    private final int smallVmRam;
    private final int smallVmStorage;
    private final int smallVmBw;
    private final int mediumVmMultiplier;
    private final int largeVmMultiplier;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final boolean payingForTheFullHour;
    private final boolean clearCreatedLists;
    private final double rewardJobWaitCoef;
    private final double rewardRunningVmCoresCoef;
    private final double rewardUnutilizedVmCoresCoef;
    private final double rewardInvalidCoef;
    private final int maxEpisodeLength;
    private final String vmAllocationPolicy;
    private final String algorithm;

    public SimulationSettings(final Map<String, Object> params) {
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
    }

    public String printSettings() {
        return "SimulationSettings {\ninitialSVmCount=" + initialSVmCount + ",\ninitialMVmCount="
                + initialMVmCount + ",\ninitialLVmCount=" + initialLVmCount + ",\nsplitLargeJobs="
                + splitLargeJobs + ",\nmaxJobPes=" + maxJobPes + ",\ntimestepInterval="
                + timestepInterval + ",\nhostPeMips=" + hostPeMips + ",\nhostBw=" + hostBw
                + ",\nhostRam=" + hostRam + ",\nhostStorage=" + hostStorage + ",\nhostPes="
                + hostPes + "\nmaxHosts=" + maxHosts + ",\nhostsCount=" + hostsCount
                + ",\nsmallVmRam=" + smallVmRam + ",\nsmallVmPes=" + smallVmPes
                + ",\nsmallVmStorage=" + smallVmStorage + ",\nsmallVmBw=" + smallVmBw
                + ",\nmediumVmMultiplier=" + mediumVmMultiplier + ",\nlargeVmMultiplier="
                + largeVmMultiplier + ",\nvmStartupDelay=" + vmStartupDelay + ",\nvmShutdownDelay="
                + vmShutdownDelay + ",\nsmallVmHourlyCost=" + smallVmHourlyCost
                + ",\npayingForTheFullHour=" + payingForTheFullHour + ",\nclearCreatedLists="
                + clearCreatedLists + ",\nrewardJobWaitCoef=" + rewardJobWaitCoef
                + ",\nrewardRunningVmCoresCoef=" + rewardRunningVmCoresCoef
                + ",\nrewardUnutilizedVmCoresCoef=" + rewardUnutilizedVmCoresCoef
                + ",\nrewardInvalidCoef=" + rewardInvalidCoef + ",\nmaxEpisodeLength="
                + maxEpisodeLength + ",\nvmAllocationPolicy=" + vmAllocationPolicy + "\nalgorithm= "
                + algorithm + ",\n}";

    }

    public int getInitialSVmCount() {
        return initialSVmCount;
    }

    public int getInitialMVmCount() {
        return initialMVmCount;
    }

    public int getInitialLVmCount() {
        return initialLVmCount;
    }

    public int[] getInitialVmCounts() {
        return initialVmCounts;
    }

    public boolean isSplitLargeJobs() {
        return splitLargeJobs;
    }

    public int getMaxJobPes() {
        return maxJobPes;
    }

    public double getTimestepInterval() {
        return timestepInterval;
    }

    public double getSmallVmHourlyCost() {
        return smallVmHourlyCost;
    }

    public long getHostPeMips() {
        return hostPeMips;
    }

    public long getHostBw() {
        return hostBw;
    }

    public long getHostRam() {
        return hostRam;
    }

    public long getHostStorage() {
        return hostStorage;
    }

    public int getHostPes() {
        return hostPes;
    }

    public int getMaxHosts() {
        return maxHosts;
    }

    public int getHostsCount() {
        return hostsCount;
    }

    public long getDatacenterCores() {
        return hostsCount * hostPes;
    }

    public int getSmallVmPes() {
        return smallVmPes;
    }

    public long getSmallVmStorage() {
        return smallVmStorage;
    }

    public long getSmallVmBw() {
        return smallVmBw;
    }

    public long getSmallVmRam() {
        return smallVmRam;
    }

    public int getMediumVmMultiplier() {
        return mediumVmMultiplier;
    }

    public int getLargeVmMultiplier() {
        return largeVmMultiplier;
    }

    public double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public double getVmShutdownDelay() {
        return vmShutdownDelay;
    }

    public long getTotalHostCores() {
        return hostsCount * hostPes;
    }

    public boolean isPayingForTheFullHour() {
        return payingForTheFullHour;
    }

    public boolean isClearCreatedLists() {
        return clearCreatedLists;
    }

    public double getRewardJobWaitCoef() {
        return rewardJobWaitCoef;
    }

    public double getRewardRunningVmCoresCoef() {
        return rewardRunningVmCoresCoef;
    }

    public double getRewardUnutilizedVmCoresCoef() {
        return rewardUnutilizedVmCoresCoef;
    }

    public double getRewardInvalidCoef() {
        return rewardInvalidCoef;
    }

    public int getMaxEpisodeLength() {
        return maxEpisodeLength;
    }

    public double getMinTimeBetweenEvents() {
        return minTimeBetweenEvents;
    }

    public String getVmAllocationPolicy() {
        return vmAllocationPolicy;
    }

    public String getAlgorithm() {
        return algorithm;
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

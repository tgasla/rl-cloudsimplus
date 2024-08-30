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

    private final double timestepInterval;
    private final int initialSVmCount;
    private final int initialMVmCount;
    private final int initialLVmCount;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final double vmHourlyCost;
    private final int hostsCount;
    private final long hostPeMips;
    private final int hostPes;
    private final long hostRam;
    private final long hostStorage;
    private final long hostBw;
    private final int smallVmPes;
    private final long smallVmRam;
    private final long smallVmStorage;
    private final long smallVmBw;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final boolean payingForTheFullHour;
    private final boolean keepCreatedCloudletList;
    private final double rewardJobWaitCoef;
    private final double rewardUtilizationCoef;
    private final double rewardInvalidCoef;
    private final int maxEpisodeLength;

    // Get SimulationSettings from environment variables
    public SimulationSettings() {
        initialSVmCount = Integer.parseInt(System.getenv("INITIAL_S_VM_COUNT"));
        initialMVmCount = Integer.parseInt(System.getenv("INITIAL_M_VM_COUNT"));
        initialLVmCount = Integer.parseInt(System.getenv("INITIAL_L_VM_COUNT"));
        timestepInterval = Double.parseDouble(System.getenv("TIMESTEP_INTERVAL"));
        splitLargeJobs = Boolean.parseBoolean(System.getenv("SPLIT_LARGE_JOBS"));
        maxJobPes = Integer.parseInt(System.getenv("MAX_JOB_PES"));
        vmHourlyCost = Double.parseDouble(System.getenv("VM_HOURLY_COST"));
        hostPeMips = Long.parseLong(System.getenv("HOST_PE_MIPS"));
        hostRam = Long.parseLong(System.getenv("HOST_RAM"));
        hostStorage = Long.parseLong(System.getenv("HOST_STORAGE"));
        hostBw = Long.parseLong(System.getenv("HOST_BW"));
        hostPes = Integer.parseInt(System.getenv("HOST_PES"));
        hostsCount = Integer.parseInt(System.getenv("HOSTS_COUNT"));
        smallVmPes = Integer.parseInt(System.getenv("SMALL_VM_PES"));
        smallVmRam = Long.parseLong(System.getenv("SMALL_VM_RAM"));
        smallVmStorage = Long.parseLong(System.getenv("SMALL_VM_STORAGE"));
        smallVmBw = Long.parseLong(System.getenv("SMALL_VM_BW"));
        vmStartupDelay = Double.parseDouble(System.getenv("VM_STARTUP_DELAY"));
        vmShutdownDelay = Double.parseDouble(System.getenv("VM_SHUTDOWN_DELAY"));
        payingForTheFullHour = Boolean.parseBoolean(System.getenv("PAYING_FOR_THE_FULL_HOUR"));
        keepCreatedCloudletList = Boolean.parseBoolean(System.getenv("KEEP_CREATED_CLOUDLET_LIST"));
        rewardJobWaitCoef = Double.parseDouble(System.getenv("REWARD_JOB_WAIT_COEF"));
        rewardUtilizationCoef = Double.parseDouble(System.getenv("REWARD_UTIL_COEF"));
        rewardInvalidCoef = Double.parseDouble(System.getenv("REWARD_INVALID_COEF"));
        maxEpisodeLength = Integer.parseInt(System.getenv("MAX_EPISODE_LENGTH"));
    }

    @Override
    public String toString() {
        return "SimulationSettings {" + "\n" + "initialSVmCount=" + initialSVmCount + ",\n"
                + "initialMVmCount=" + initialMVmCount + ",\n" + "initialLVmCount="
                + initialLVmCount + ",\n" + "splitLargeJobs=" + splitLargeJobs + ",\n"
                + "maxJobPes=" + maxJobPes + ",\n" + "timestepInterval=" + timestepInterval + ",\n"
                + "hostPeMips=" + hostPeMips + ",\n" + "hostBw=" + hostBw + ",\n" + "hostRam="
                + hostRam + ",\n" + "hostStorage=" + hostStorage + ",\n" + "hostPes=" + hostPes
                + ",\n" + "hostsCount=" + hostsCount + ",\n" + "smallVmRam=" + smallVmRam + ",\n"
                + "smallVmPes=" + smallVmPes + ",\n" + "smallVmStorage=" + smallVmStorage + ",\n"
                + "smallVmBw=" + smallVmBw + ",\n" + "vmStartupDelay" + vmStartupDelay + ",\n"
                + "vmShutdownDelay" + vmShutdownDelay + ",\n" + "vmHourlyCost=" + vmHourlyCost
                + ",\n" + "payingForTheFullHour=" + payingForTheFullHour + ",\n"
                + "keepCreatedCloudletList=" + keepCreatedCloudletList + ",\n"
                + "rewardJobWaitCoef=" + rewardJobWaitCoef + ",\n" + "rewardUtilizationCoef="
                + rewardUtilizationCoef + ",\n" + "rewardInvalidCoef=" + rewardInvalidCoef + ",\n"
                + "maxEpisodeLength=" + maxEpisodeLength + ",\n" + "}";
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

    public boolean isSplitLargeJobs() {
        return splitLargeJobs;
    }

    public int getMaxJobPes() {
        return maxJobPes;
    }

    public double getTimestepInterval() {
        return timestepInterval;
    }

    public double getVmHourlyCost() {
        return vmHourlyCost;
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

    public boolean isKeepCreatedCloudletList() {
        return keepCreatedCloudletList;
    }

    public double getRewardJobWaitCoef() {
        return rewardJobWaitCoef;
    }

    public double getRewardUtilizationCoef() {
        return rewardUtilizationCoef;
    }

    public double getRewardInvalidCoef() {
        return rewardInvalidCoef;
    }

    public int getMaxEpisodeLength() {
        return maxEpisodeLength;
    }
}

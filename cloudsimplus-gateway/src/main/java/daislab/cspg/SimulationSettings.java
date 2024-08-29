package daislab.cspg;

import java.util.Map;

// import static org.apache.commons.lang3.SystemUtils.getEnvironmentVariable;

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
    private final String jobs;
    private final String sourceOfJobs;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final double vmRunningHourlyCost;
    private final long hostPeMips;
    private final long hostBw;
    private final long hostRam;
    private final long hostSize;
    private final int hostPeCnt;
    private final int datacenterHostsCnt;
    private final long basicVmRam;
    private final int basicVmPeCount;
    private final long basicVmSize;
    private final long basicVmBw;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final boolean printJobsPeriodically;
    private final boolean payingForTheFullHour;
    private final boolean storeCreatedCloudletsDatacenterBroker;
    private final double rewardJobWaitCoef;
    private final double rewardUtilizationCoef;
    private final double rewardInvalidCoef;
    private final String jobLogDir;
    private final int maxSteps;

    // Get SimulationSettings from parameters
    // passed from the python client endpoint - the Gymnasium environment,
    // if an environment variable is not set, a default value is given
    public SimulationSettings(final Map<String, String> parameters) {
        sourceOfJobs = parameters.getOrDefault("SOURCE_OF_JOBS", "PARAMS");
        jobs = parameters.getOrDefault("JOBS", "[]");
        initialSVmCount = Integer.parseInt(parameters.getOrDefault("INITIAL_S_VM_COUNT", "0"));
        initialMVmCount = Integer.parseInt(parameters.getOrDefault("INITIAL_M_VM_COUNT", "0"));
        initialLVmCount = Integer.parseInt(parameters.getOrDefault("INITIAL_L_VM_COUNT", "0"));
        timestepInterval = Double.parseDouble(parameters.getOrDefault("TIMESTEP_INTERVAL", "1.0"));
        splitLargeJobs = Boolean
                .parseBoolean(parameters.getOrDefault("SPLIT_LARGE_JOBS", "true").toLowerCase());
        maxJobPes = Integer.parseInt(parameters.getOrDefault("MAX_JOB_PES", "1"));
        vmRunningHourlyCost =
                Double.parseDouble(parameters.getOrDefault("VM_RUNNING_HOURLY_COST", "0.086"));
        hostPeMips = Long.parseLong(parameters.getOrDefault("HOST_PE_MIPS", "10000"));
        hostBw = Long.parseLong(parameters.getOrDefault("HOST_BW", "50000"));
        hostRam = Long.parseLong(parameters.getOrDefault("HOST_RAM", "65536"));
        hostSize = Long.parseLong(parameters.getOrDefault("HOST_SIZE", "100000"));
        hostPeCnt = Integer.parseInt(parameters.getOrDefault("HOST_PE_CNT", "14"));
        datacenterHostsCnt =
                Integer.parseInt(parameters.getOrDefault("DATACENTER_HOSTS_CNT", "3000"));
        basicVmRam = Long.parseLong(parameters.getOrDefault("BASIC_VM_RAM", "8192"));
        basicVmPeCount = Integer.parseInt(parameters.getOrDefault("BASIC_VM_PE_CNT", "2"));
        basicVmSize = Long.parseLong(parameters.getOrDefault("BASIC_VM_SIZE", "4000"));
        basicVmBw = Long.parseLong(parameters.getOrDefault("BASIC_VM_BW", "1000"));
        vmStartupDelay = Double.parseDouble(parameters.getOrDefault("VM_STARTUP_DELAY", "0"));
        vmShutdownDelay = Double.parseDouble(parameters.getOrDefault("VM_SHUTDOWN_DELAY", "0"));
        printJobsPeriodically =
                Boolean.parseBoolean(parameters.getOrDefault("PRINT_JOBS_PERIODICALLY", "true"));
        payingForTheFullHour =
                Boolean.parseBoolean(parameters.getOrDefault("PAYING_FOR_THE_FULL_HOUR", "false"));
        storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(
                parameters.getOrDefault("STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));
        rewardJobWaitCoef =
                Double.parseDouble(parameters.getOrDefault("REWARD_JOB_WAIT_COEF", "0.3"));
        rewardUtilizationCoef =
                Double.parseDouble(parameters.getOrDefault("REWARD_UTILIZATION_COEF", "0.3"));
        rewardInvalidCoef =
                Double.parseDouble(parameters.getOrDefault("REWARD_INVALID_COEF", "0.4"));
        jobLogDir = parameters.getOrDefault("JOB_LOG_DIR", "./logs");
        maxSteps = Integer.parseInt(parameters.getOrDefault("MAX_TIMESTEPS_PER_EPISODE", "5000"));
    }

    @Override
    public String toString() {
        return "SimulationSettings {" + "\n" + "vmRunningHourlyCost=" + vmRunningHourlyCost + ",\n"
                + "initialSVmCount=" + initialSVmCount + ",\n" + "initialMVmCount="
                + initialMVmCount + ",\n" + "initialLVmCount=" + initialLVmCount + ",\n"
                + "sourceOfJobs=" + sourceOfJobs + ",\n" + "splitLargeJobs=" + splitLargeJobs
                + ",\n" + "maxJobPes=" + maxJobPes + ",\n" + "timestepInterval=" + timestepInterval
                + ",\n" + "hostPeMips=" + hostPeMips + ",\n" + "hostBw=" + hostBw + ",\n"
                + "hostRam=" + hostRam + ",\n" + "hostSize=" + hostSize + ",\n" + "hostPeCnt="
                + hostPeCnt + ",\n" + "datacenterHostsCnt=" + datacenterHostsCnt + ",\n"
                + "basicVmRam=" + basicVmRam + ",\n" + "basicVmPeCount=" + basicVmPeCount + ",\n"
                + "basicVmSize=" + basicVmSize + ",\n" + "basicVmBw=" + basicVmBw + ",\n"
                + "vmStartupDelay" + vmStartupDelay + ",\n" + "vmShutdownDelay" + vmShutdownDelay
                + ",\n" + "printJobsPeriodically=" + printJobsPeriodically + ",\n"
                + "payingForTheFullHour=" + payingForTheFullHour + ",\n"
                + "storeCreatedCloudletsDatacenterBroker=" + storeCreatedCloudletsDatacenterBroker
                + ",\n" + "rewardJobWaitCoef=" + rewardJobWaitCoef + ",\n"
                + "rewardUtilizationCoef=" + rewardUtilizationCoef + ",\n" + "rewardInvalidCoef="
                + rewardInvalidCoef + ",\n" + "jobLogDir=" + jobLogDir + ",\n" + "maxSteps="
                + maxSteps + ",\n" + "}";
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

    public String getSourceOfJobs() {
        return sourceOfJobs;
    }

    public String getJobsAsJson() {
        return jobs;
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

    public double getVmRunningHourlyCost() {
        return vmRunningHourlyCost;
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

    public long getHostSize() {
        return hostSize;
    }

    public int getHostPeCnt() {
        return hostPeCnt;
    }

    public int getDatacenterHostsCnt() {
        return datacenterHostsCnt;
    }

    public long getDatacenterCores() {
        return datacenterHostsCnt * hostPeCnt;
    }

    public int getBasicVmPeCnt() {
        return basicVmPeCount;
    }

    public double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public double getVmShutdownDelay() {
        return vmShutdownDelay;
    }

    public long getBasicVmSize() {
        return basicVmSize;
    }

    public long getBasicVmBw() {
        return basicVmBw;
    }

    public long getBasicVmRam() {
        return basicVmRam;
    }

    public long getTotalHostCores() {
        return datacenterHostsCnt * hostPeCnt;
    }

    public boolean isPrintJobsPeriodically() {
        return printJobsPeriodically;
    }

    public boolean isPayingForTheFullHour() {
        return payingForTheFullHour;
    }

    public boolean isStoreCreatedCloudletsDatacenterBroker() {
        return storeCreatedCloudletsDatacenterBroker;
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

    public String getJobLogDir() {
        return jobLogDir;
    }

    public int getMaxSteps() {
        return maxSteps;
    }
}

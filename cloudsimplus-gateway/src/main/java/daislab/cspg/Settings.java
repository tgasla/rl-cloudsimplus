package daislab.cspg;

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
public class Settings {
    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};
    private static final double minTimeBetweenEvents = 0.1;

    private static final double timestepInterval =
            Double.parseDouble(System.getenv("TIMESTEP_INTERVAL"));
    private static final int initialSVmCount =
            Integer.parseInt(System.getenv("INITIAL_S_VM_COUNT"));
    private static final int initialMVmCount =
            Integer.parseInt(System.getenv("INITIAL_M_VM_COUNT"));
    private static final int initialLVmCount =
            Integer.parseInt(System.getenv("INITIAL_L_VM_COUNT"));
    private static final boolean splitLargeJobs =
            Boolean.parseBoolean(System.getenv("SPLIT_LARGE_JOBS"));
    private static final int maxJobPes = Integer.parseInt(System.getenv("MAX_JOB_PES"));
    private static final double vmHourlyCost = Double.parseDouble(System.getenv("VM_HOURLY_COST"));
    private static final int hostsCount = Integer.parseInt(System.getenv("HOST_COUNT"));
    private static final long hostPeMips = Long.parseLong(System.getenv("HOST_PE_MIPS"));
    private static final int hostPes = Integer.parseInt(System.getenv("HOST_PES"));
    private static final long hostRam = Long.parseLong(System.getenv("HOST_RAM"));
    private static final long hostStorage = Long.parseLong(System.getenv("HOST_STORAGE"));
    private static final long hostBw = Long.parseLong(System.getenv("HOST_BW"));
    private static final int smallVmPes = Integer.parseInt(System.getenv("SMALL_VM_PES"));
    private static final long smallVmRam = Long.parseLong(System.getenv("SMALL_VM_RAM"));
    private static final long smallVmStorage = Long.parseLong(System.getenv("SMALL_VM_STORAGE"));
    private static final long smallVmBw = Long.parseLong(System.getenv("SMALL_VM_BW"));
    private static final int mediumVmMultiplier =
            Integer.parseInt(System.getenv("MEDIUM_VM_MULTIPLIER"));
    private static final int largeVmMultiplier =
            Integer.parseInt(System.getenv("LARGE_VM_MULTIPLIER"));
    private static final double vmStartupDelay =
            Double.parseDouble(System.getenv("VM_STARTUP_DELAY"));
    private static final double vmShutdownDelay =
            Double.parseDouble(System.getenv("VM_SHUTDOWN_DELAY"));
    private static final boolean payingForTheFullHour =
            Boolean.parseBoolean(System.getenv("PAYING_FOR_THE_FULL_HOUR"));
    private static final boolean clearCreatedCloudletList =
            Boolean.parseBoolean(System.getenv("CLEAR_CREATED_CLOUDLET_LIST"));
    private static final double rewardJobWaitCoef =
            Double.parseDouble(System.getenv("REWARD_JOB_WAIT_COEF"));
    private static final double rewardRunningVmCoresCoef =
            Double.parseDouble(System.getenv("REWARD_RUNNING_VM_CORES_COEF"));
    private static final double rewardUnutilizedVmCoresCoef =
            Double.parseDouble(System.getenv("REWARD_UNUTILIZED_VM_CORES_COEF"));
    private static final double rewardInvalidCoef =
            Double.parseDouble(System.getenv("REWARD_INVALID_COEF"));
    private static final int maxEpisodeLength =
            Integer.parseInt(System.getenv("MAX_EPISODE_LENGTH"));

    public static String printSettings() {
        return "SimulationSettings {\ninitialSVmCount=" + initialSVmCount + ",\ninitialMVmCount="
                + initialMVmCount + ",\ninitialLVmCount=" + initialLVmCount + ",\nsplitLargeJobs="
                + splitLargeJobs + ",\nmaxJobPes=" + maxJobPes + ",\ntimestepInterval="
                + timestepInterval + ",\nhostPeMips=" + hostPeMips + ",\nhostBw=" + hostBw + ",\n"
                + "hostRam=" + hostRam + ",\nhostStorage=" + hostStorage + ",\nhostPes=" + hostPes
                + ",\nhostsCount=" + hostsCount + ",\nsmallVmRam=" + smallVmRam + ",\nsmallVmPes="
                + smallVmPes + ",\nsmallVmStorage=" + smallVmStorage + ",\nsmallVmBw=" + smallVmBw
                + ",\nmediumVmMultiplier=" + mediumVmMultiplier + ",\nlargeVmMultiplier="
                + largeVmMultiplier + ",\nvmStartupDelay=" + vmStartupDelay + ",\nvmShutdownDelay="
                + vmShutdownDelay + ",\nvmHourlyCost=" + vmHourlyCost + ",\npayingForTheFullHour="
                + payingForTheFullHour + ",\n" + "clearCreatedCloudletList="
                + clearCreatedCloudletList + ",\nrewardJobWaitCoef=" + rewardJobWaitCoef
                + ",\nrewardRunningVmCoresCoef=" + rewardRunningVmCoresCoef
                + ",\nrewardUnutilizedVmCoresCoef=" + rewardUnutilizedVmCoresCoef
                + ",\nrewardInvalidCoef=" + rewardInvalidCoef + ",\nmaxEpisodeLength="
                + maxEpisodeLength + ",\n}";
    }

    public static int getInitialSVmCount() {
        return initialSVmCount;
    }

    public static int getInitialMVmCount() {
        return initialMVmCount;
    }

    public static int getInitialLVmCount() {
        return initialLVmCount;
    }

    public static boolean isSplitLargeJobs() {
        return splitLargeJobs;
    }

    public static int getMaxJobPes() {
        return maxJobPes;
    }

    public static double getTimestepInterval() {
        return timestepInterval;
    }

    public static double getVmHourlyCost() {
        return vmHourlyCost;
    }

    public static long getHostPeMips() {
        return hostPeMips;
    }

    public static long getHostBw() {
        return hostBw;
    }

    public static long getHostRam() {
        return hostRam;
    }

    public static long getHostStorage() {
        return hostStorage;
    }

    public static int getHostPes() {
        return hostPes;
    }

    public static int getHostsCount() {
        return hostsCount;
    }

    public static long getDatacenterCores() {
        return hostsCount * hostPes;
    }

    public static int getSmallVmPes() {
        return smallVmPes;
    }

    public static long getSmallVmStorage() {
        return smallVmStorage;
    }

    public static long getSmallVmBw() {
        return smallVmBw;
    }

    public static long getSmallVmRam() {
        return smallVmRam;
    }

    public static int getMediumVmMultiplier() {
        return mediumVmMultiplier;
    }

    public static int getLargeVmMultiplier() {
        return largeVmMultiplier;
    }

    public static double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public static double getVmShutdownDelay() {
        return vmShutdownDelay;
    }

    public static long getTotalHostCores() {
        return hostsCount * hostPes;
    }

    public static boolean isPayingForTheFullHour() {
        return payingForTheFullHour;
    }

    public static boolean isClearCreatedCloudletList() {
        return clearCreatedCloudletList;
    }

    public static double getRewardJobWaitCoef() {
        return rewardJobWaitCoef;
    }

    public static double getRewardRunningVmCoresCoef() {
        return rewardRunningVmCoresCoef;
    }

    public static double getRewardUnutilizedVmCoresCoef() {
        return rewardUnutilizedVmCoresCoef;
    }

    public static double getRewardInvalidCoef() {
        return rewardInvalidCoef;
    }

    public static int getMaxEpisodeLength() {
        return maxEpisodeLength;
    }

    public static double getMinTimeBetweenEvents() {
        return minTimeBetweenEvents;
    }

    public static int getSizeMultiplier(final String type) {
        switch (type) {
            case MEDIUM:
                return mediumVmMultiplier; // m5a.xlarge
            case LARGE:
                return largeVmMultiplier; // m5a.2xlarge
            case SMALL:
            default:
                return 1; // m5a.large
        }
    }
}

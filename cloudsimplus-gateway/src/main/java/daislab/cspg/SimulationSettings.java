package daislab.cspg;

import java.util.Map;
import lombok.Data;
import java.util.List;

/*
 * Class to describe the simulation settings.
 *
 * It takes as a parameter a Map<String, String>. The first string represents the parameter name and
 * the second string represents the parameter value.
 */
@Data
public class SimulationSettings {
    private final String runMode;
    private final int numExperiments;
    private final double minTimeBetweenEvents;
    private final double timestepInterval;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final int maxHosts;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final boolean payingForTheFullHour;
    private final boolean clearCreatedLists;
    private final double rewardJobsPlacedCoef;
    private final double rewardQualityCoef;
    private final double rewardDeadlineViolationCoef;
    private final int maxEpisodeLength;
    private final String vmAllocationPolicy;
    private final String cloudletToDcAssignmentPolicy;
    private final String cloudletToVmAssignmentPolicy;
    private final String algorithm;
    private final String stateActionSpaceType;
    private final int maxJobsWaiting;
    private final List<Map<String, Object>> datacenters;
    // private final double rewardJobWaitCoef;
    // private final double rewardRunningVmCoresCoef;
    // private final double rewardUnutilizedVmCoresCoef;
    // private final double rewardInvalidCoef;
}

package daislab.cspg;

import java.util.Map;
import lombok.Data;
import java.util.List;

/*
 * Class to describe the simulation settings.
 *
 * It takes as a parameter a Map<String, Object>. Parameters are accessed via
 * safe Number-aware getters to handle Gson LazilyParsedNumber correctly.
 */
@Data
public class SimulationSettings implements ISimulationSettings {
    private final String mode;
    private final int numExperiments;
    private final double minTimeBetweenEvents;
    private final double timestepInterval;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final int maxEpisodeLength;
    private final String algorithm;
    private final String cloudletToDcAssignmentPolicy;
    private final String cloudletToVmAssignmentPolicy;
    private final String stateSpaceType;
    private final String vmAllocationPolicy;
    private final int maxJobsWaiting;
    private final List<Map<String, Object>> datacenters;
    private final int maxHosts;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final boolean payingForTheFullHour;
    private final boolean clearCreatedLists;
    private final boolean printStats;
    private final double rewardJobsPlacedCoef;
    private final double rewardQualityCoef;
    private final double rewardDeadlineViolationCoef;

    public SimulationSettings(final Map<String, Object> params) {
        mode = ISimulationSettings.getStr(params, "mode");
        numExperiments = ISimulationSettings.getInt(params, "num_experiments");
        minTimeBetweenEvents = ISimulationSettings.getDouble(params, "min_time_between_events");
        timestepInterval = ISimulationSettings.getDouble(params, "timestep_interval");
        splitLargeJobs = ISimulationSettings.getBool(params, "split_large_jobs");
        maxJobPes = ISimulationSettings.getInt(params, "max_job_pes");
        maxHosts = ISimulationSettings.getInt(params, "max_hosts");
        vmStartupDelay = ISimulationSettings.getDouble(params, "vm_startup_delay");
        vmShutdownDelay = ISimulationSettings.getDouble(params, "vm_shutdown_delay");
        payingForTheFullHour = ISimulationSettings.getBool(params, "paying_for_the_full_hour");
        clearCreatedLists = ISimulationSettings.getBool(params, "clear_created_lists");
        printStats = ISimulationSettings.getBoolOrDefault(params, "print_stats", true);
        rewardJobsPlacedCoef = ISimulationSettings.getDouble(params, "reward_jobs_placed_coef");
        rewardQualityCoef = ISimulationSettings.getDouble(params, "reward_quality_coef");
        rewardDeadlineViolationCoef = ISimulationSettings.getDouble(params, "reward_deadline_violation_coef");
        maxEpisodeLength = ISimulationSettings.getInt(params, "max_episode_length");
        algorithm = ISimulationSettings.getStr(params, "algorithm");
        cloudletToDcAssignmentPolicy = ISimulationSettings.getStr(params, "cloudlet_to_dc_assignment_policy");
        cloudletToVmAssignmentPolicy = ISimulationSettings.getStr(params, "cloudlet_to_vm_assignment_policy");
        stateSpaceType = ISimulationSettings.getStr(params, "state_space_type");
        vmAllocationPolicy = ISimulationSettings.getStr(params, "vm_allocation_policy");
        maxJobsWaiting = ISimulationSettings.getInt(params, "max_jobs_waiting");
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
    }

    @Override
    public Map<String, Object> getParams() {
        return Map.of();
    }
}
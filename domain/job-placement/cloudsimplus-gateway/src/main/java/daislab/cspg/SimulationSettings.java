package daislab.cspg;

import java.util.Map;
import lombok.Value;
import java.util.List;

/*
 * Class to describe the simulation settings.
 *
 * It takes as a parameter a Map<String, Object>. Parameters are accessed via
 * safe Number-aware getters to handle Gson LazilyParsedNumber correctly.
 */
@Value
public class SimulationSettings implements ISimulationSettings {
    private final String mode;
    private final int numExperiments;
    private final double minTimeBetweenEvents;
    private final double timestepInterval;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final int maxEpisodeLength;
    private final String rlAlgorithm;
    private final String cloudletToDcMapping;
    private final String cloudletToVmMapping;
    private final String stateSpaceType;
    private final int maxJobsWaiting;
    private final List<Map<String, Object>> datacenters;
    private final int maxHosts;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final int driftAtStep;
    private final int driftDcIndex;
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
        driftAtStep = ISimulationSettings.getIntOrDefault(params, "drift_at_step", -1);
        driftDcIndex = ISimulationSettings.getIntOrDefault(params, "drift_dc_index", 0);
        clearCreatedLists = ISimulationSettings.getBool(params, "clear_created_lists");
        printStats = ISimulationSettings.getBoolOrDefault(params, "print_stats", true);
        rewardJobsPlacedCoef = ISimulationSettings.getDouble(params, "reward_jobs_placed_coef");
        rewardQualityCoef = ISimulationSettings.getDouble(params, "reward_quality_coef");
        rewardDeadlineViolationCoef = ISimulationSettings.getDouble(params, "reward_deadline_violation_coef");
        maxEpisodeLength = ISimulationSettings.getInt(params, "max_episode_length");
        rlAlgorithm = ISimulationSettings.getStr(params, "rl_algorithm");
        cloudletToDcMapping = ISimulationSettings.getStr(params, "cloudlet_to_dc_mapping");
        cloudletToVmMapping = ISimulationSettings.getStr(params, "cloudlet_to_vm_mapping");
        stateSpaceType = ISimulationSettings.getStr(params, "state_space_type");
        maxJobsWaiting = ISimulationSettings.getInt(params, "max_jobs_waiting");
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
    }

}
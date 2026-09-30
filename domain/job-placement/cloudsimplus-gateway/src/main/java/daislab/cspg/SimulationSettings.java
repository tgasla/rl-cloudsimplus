package daislab.cspg;

import java.util.HashMap;
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
    static final List<String> SENSITIVITIES = List.of("tolerant", "moderate", "critical");
    static final List<String> DC_TYPES = List.of("cloud", "edge", "micro");
    // CloudSim advances a running cloudlet by a whole number of instructions per update and
    // drops the fraction, so recorded finishes drift late on busy DCs (seconds, per episode).
    // Inside the simulator one MI of workload is counted as MI_RESOLUTION instructions, and
    // every PE runs MI_RESOLUTION times as many per second: all times and ratios are unchanged.
    static final double MI_RESOLUTION = 1000;

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
    // PE speed that defines one unit of nominal runtime (the edge tier in RING-N).
    private final double mipsRef;
    // Net-SLA reward, indexed by delay sensitivity (tolerant, moderate, critical): value earned
    // when a job finishes by its due time, penalty paid when it does not.
    private final double[] slaValue;
    private final double[] slaPenalty;
    // Per DC type: price per reference core-second, and network delay in timesteps.
    private final Map<String, Double> costPerRefCoreSecond;
    private final Map<String, Double> networkDelayTimesteps;
    // Potential-based shaping; its gamma must equal the RL algorithm's (checked in Python).
    private final boolean rewardShaping;
    private final double rewardShapingGamma;

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
        maxEpisodeLength = ISimulationSettings.getInt(params, "max_episode_length");
        rlAlgorithm = ISimulationSettings.getStr(params, "rl_algorithm");
        cloudletToDcMapping = ISimulationSettings.getStr(params, "cloudlet_to_dc_mapping");
        cloudletToVmMapping = ISimulationSettings.getStr(params, "cloudlet_to_vm_mapping");
        stateSpaceType = ISimulationSettings.getStr(params, "state_space_type");
        maxJobsWaiting = ISimulationSettings.getInt(params, "max_jobs_waiting");
        mipsRef = ISimulationSettings.getDouble(params, "mips_ref") * MI_RESOLUTION;
        // Flat scalars: JSON arrays inside params do not survive SimulationFactoryBase's parsing.
        slaValue = new double[SENSITIVITIES.size()];
        slaPenalty = new double[SENSITIVITIES.size()];
        for (int s = 0; s < SENSITIVITIES.size(); s++) {
            slaValue[s] = ISimulationSettings.getDouble(params, "sla_value_" + SENSITIVITIES.get(s));
            slaPenalty[s] = ISimulationSettings.getDouble(params, "sla_penalty_" + SENSITIVITIES.get(s));
        }
        costPerRefCoreSecond = new HashMap<>();
        networkDelayTimesteps = new HashMap<>();
        for (String dcType : DC_TYPES) {
            costPerRefCoreSecond.put(dcType, ISimulationSettings.getDouble(params, "cost_" + dcType));
            networkDelayTimesteps.put(dcType,
                    ISimulationSettings.getDouble(params, "network_delay_" + dcType));
        }
        rewardShaping = ISimulationSettings.getBool(params, "reward_shaping");
        rewardShapingGamma = ISimulationSettings.getDouble(params, "reward_shaping_gamma");
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
    }


    double slaValue(final int sensitivity) {
        return slaValue[sensitivity];
    }

    double slaPenalty(final int sensitivity) {
        return slaPenalty[sensitivity];
    }

    double costPerRefCoreSecond(final String dcType) {
        return costPerRefCoreSecond.get(dcType);
    }

    /** Network delay of a DC type, in simulation seconds. */
    double networkDelay(final String dcType) {
        return networkDelayTimesteps.get(dcType) * timestepInterval;
    }
}

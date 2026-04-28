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
    private final String stateSpaceType;
    private final int maxJobsWaiting;
    private final List<Map<String, Object>> datacenters;

    public SimulationSettings(final Map<String, Object> params) {
        runMode = getStr(params, "run_mode", "train");
        numExperiments = getInt(params, "num_experiments", 1);
        minTimeBetweenEvents = getDouble(params, "min_time_between_events", 0.1);
        timestepInterval = getDouble(params, "timestep_interval", 1.0);
        splitLargeJobs = getBool(params, "split_large_jobs", false);
        maxJobPes = getInt(params, "max_job_pes", 16);
        maxHosts = getInt(params, "max_hosts", 40);
        vmStartupDelay = getDouble(params, "vm_startup_delay", 0.0);
        vmShutdownDelay = getDouble(params, "vm_shutdown_delay", 0.0);
        payingForTheFullHour = getBool(params, "paying_for_the_full_hour", false);
        clearCreatedLists = getBool(params, "clear_created_lists", false);
        rewardJobsPlacedCoef = getDouble(params, "reward_jobs_placed_coef", 0.333);
        rewardQualityCoef = getDouble(params, "reward_quality_coef", 0.333);
        rewardDeadlineViolationCoef = getDouble(params, "reward_deadline_violation_coef", 0.333);
        maxEpisodeLength = getInt(params, "max_episode_length", 150);
        vmAllocationPolicy = getStr(params, "vm_allocation_policy", "rl");
        cloudletToDcAssignmentPolicy = getStr(params, "cloudlet_to_dc_assignment_policy", "rl");
        cloudletToVmAssignmentPolicy = getStr(params, "cloudlet_to_vm_assignment_policy", "most-free-cores");
        algorithm = getStr(params, "algorithm", "PPO");
        stateSpaceType = getStr(params, "state_space_type", "dcid-dctype-freevmpes-per-host");
        maxJobsWaiting = getInt(params, "max_jobs_waiting", 50);
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
    }

    private static int getInt(Map<String, Object> m, String k, int def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(v.toString()); } catch (Exception e) { return def; }
    }
    private static double getDouble(Map<String, Object> m, String k, double def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); } catch (Exception e) { return def; }
    }
    private static boolean getBool(Map<String, Object> m, String k, boolean def) {
        Object v = m.get(k);
        if (v == null) return def;
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(v.toString());
    }
    private static String getStr(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        return v == null ? def : v.toString();
    }
}
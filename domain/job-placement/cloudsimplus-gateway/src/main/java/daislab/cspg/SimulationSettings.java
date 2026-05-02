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
    private final double rewardJobsPlacedCoef;
    private final double rewardQualityCoef;
    private final double rewardDeadlineViolationCoef;

    public SimulationSettings(final Map<String, Object> params) {
        mode = getStr(params, "mode");
        numExperiments = getInt(params, "num_experiments");
        minTimeBetweenEvents = getDouble(params, "min_time_between_events");
        timestepInterval = getDouble(params, "timestep_interval");
        splitLargeJobs = getBool(params, "split_large_jobs");
        maxJobPes = getInt(params, "max_job_pes");
        maxHosts = getInt(params, "max_hosts");
        vmStartupDelay = getDouble(params, "vm_startup_delay");
        vmShutdownDelay = getDouble(params, "vm_shutdown_delay");
        payingForTheFullHour = getBool(params, "paying_for_the_full_hour");
        clearCreatedLists = getBool(params, "clear_created_lists");
        rewardJobsPlacedCoef = getDouble(params, "reward_jobs_placed_coef");
        rewardQualityCoef = getDouble(params, "reward_quality_coef");
        rewardDeadlineViolationCoef = getDouble(params, "reward_deadline_violation_coef");
        maxEpisodeLength = getInt(params, "max_episode_length");
        algorithm = getStr(params, "algorithm");
        cloudletToDcAssignmentPolicy = getStr(params, "cloudlet_to_dc_assignment_policy");
        cloudletToVmAssignmentPolicy = getStr(params, "cloudlet_to_vm_assignment_policy");
        stateSpaceType = getStr(params, "state_space_type");
        vmAllocationPolicy = getStr(params, "vm_allocation_policy");
        maxJobsWaiting = getInt(params, "max_jobs_waiting");
        datacenters = (List<Map<String, Object>>) getOrNull(params, "datacenters");
    }

    @Override
    public Map<String, Object> getParams() {
        return Map.of();
    }

    private static Object getOrNull(Map<String, Object> m, String k) {
        return m.get(k);
    }

    private static int getInt(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required int parameter: " + k);
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse int parameter '" + k + "': " + v);
        }
    }
    private static double getDouble(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required double parameter: " + k);
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse double parameter '" + k + "': " + v);
        }
    }
    private static boolean getBool(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required boolean parameter: " + k);
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(v.toString());
    }
    private static String getStr(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required string parameter: " + k);
        return v.toString();
    }
}
package daislab.cspg;

import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/*
 * Class to describe the simulation settings.
 *
 * It takes as a parameter a Map<String, String>. The first string represents the parameter name and
 * the second string represents the parameter value.
 */
public class SimulationSettings {
    private final double minTimeBetweenEvents = 0.1;
    private final double timestepInterval;
    private final boolean splitLargeJobs;
    private final int maxJobPes;
    private final int maxHosts;
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
    private final List<Map<String, Object>> datacenterListOfMaps;

    public SimulationSettings(final Map<String, Object> params) {
        timestepInterval = (double) params.get("timestep_interval");
        splitLargeJobs = (boolean) params.get("split_large_jobs");
        maxJobPes = (int) params.get("max_job_pes");
        maxHosts = (int) params.get("max_hosts");
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
        datacenterListOfMaps = castToListOfMapStringObject(params.get("datacenters"));
    }

    List<Map<String, Object>> castToListOfMapStringObject(Object obj) {
        if (obj instanceof List<?>) {
            List<?> rawList = (List<?>) obj;
            List<Map<String, Object>> safeList = new ArrayList<>();

            for (Object item : rawList) {
                if (item instanceof Map<?, ?>) {
                    Map<?, ?> rawMap = (Map<?, ?>) item;

                    // Create a new Map<String, String> after type-checking
                    Map<String, Object> safeMap = new HashMap<>();
                    for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
                        System.out.println(entry.getKey().getClass());
                        if (entry.getKey() instanceof String) {
                            safeMap.put((String) entry.getKey(), entry.getValue());
                            System.out.println(entry.getKey() + " " + entry.getValue().toString());
                        } else {
                            throw new IllegalArgumentException("Map contains non-string keys");
                        }
                    }
                    safeList.add(safeMap);
                } else {
                    throw new IllegalArgumentException("List contains non-map elements");
                }
            }
            return safeList;
        } else {
            throw new IllegalArgumentException("Object is not a List");
        }
    }

    public String printSettings() {
        return "SimulationSettings {\ninitialSVmCount=" + ",\nsplitLargeJobs=" + splitLargeJobs
                + ",\nmaxJobPes=" + maxJobPes + ",\ntimestepInterval=" + timestepInterval
                + "\nmaxHosts=" + maxHosts + ",\nvmStartupDelay=" + vmStartupDelay
                + ",\nvmShutdownDelay=" + vmShutdownDelay + ",\npayingForTheFullHour="
                + payingForTheFullHour + ",\nclearCreatedLists=" + clearCreatedLists
                + ",\nrewardJobWaitCoef=" + rewardJobWaitCoef + ",\nrewardRunningVmCoresCoef="
                + rewardRunningVmCoresCoef + ",\nrewardUnutilizedVmCoresCoef="
                + rewardUnutilizedVmCoresCoef + ",\nrewardInvalidCoef=" + rewardInvalidCoef
                + ",\nmaxEpisodeLength=" + maxEpisodeLength + ",\nvmAllocationPolicy="
                + vmAllocationPolicy + "\nalgorithm= " + algorithm + ",\n}";
    }

    public List<Map<String, Object>> getDatacenterListOfMaps() {
        return datacenterListOfMaps;
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

    public int getMaxHosts() {
        return maxHosts;
    }

    public double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public double getVmShutdownDelay() {
        return vmShutdownDelay;
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
}

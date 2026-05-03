package daislab.cspg;

import lombok.Value;
import java.util.List;
import java.util.Map;

@Value
public class SimulationSettings implements ISimulationSettings {

    Map<String, Object> params;

    double minTimeBetweenEvents;
    double timestepInterval;
    boolean splitLargeJobs;
    int maxJobPes;
    int maxHosts;
    double smallVmHourlyCost;
    double vmStartupDelay;
    double vmShutdownDelay;
    boolean payingForTheFullHour;
    boolean clearCreatedLists;
    double rewardJobWaitCoef;
    double rewardRunningVmCoresCoef;
    double rewardUnutilizedVmCoresCoef;
    double rewardInvalidCoef;
    int maxEpisodeLength;
    String vmAllocationPolicy;
    String algorithm;
    boolean sendObservationTreeArray;
    List<Map<String, Object>> datacenters;
    List<Map<String, Object>> vmTypes;

    // Derived from topology — computed once in constructor
    int hostsCount;
    long datacenterCores;

    @SuppressWarnings("unchecked")
    public SimulationSettings(final Map<String, Object> params) {
        this.params = params;
        minTimeBetweenEvents = ISimulationSettings.getDouble(params, "min_time_between_events");
        timestepInterval = ISimulationSettings.getDouble(params, "timestep_interval");
        splitLargeJobs = ISimulationSettings.getBool(params, "split_large_jobs");
        maxJobPes = ISimulationSettings.getInt(params, "max_job_pes");
        maxHosts = ISimulationSettings.getInt(params, "max_hosts");
        smallVmHourlyCost = ISimulationSettings.getDouble(params, "small_vm_hourly_cost");
        vmStartupDelay = ISimulationSettings.getDouble(params, "vm_startup_delay");
        vmShutdownDelay = ISimulationSettings.getDouble(params, "vm_shutdown_delay");
        payingForTheFullHour = ISimulationSettings.getBool(params, "paying_for_the_full_hour");
        clearCreatedLists = ISimulationSettings.getBool(params, "clear_created_lists");
        rewardJobWaitCoef = ISimulationSettings.getDouble(params, "reward_job_wait_coef");
        rewardRunningVmCoresCoef = ISimulationSettings.getDouble(params, "reward_running_vm_cores_coef");
        rewardUnutilizedVmCoresCoef = ISimulationSettings.getDouble(params, "reward_unutilized_vm_cores_coef");
        rewardInvalidCoef = ISimulationSettings.getDouble(params, "reward_invalid_coef");
        maxEpisodeLength = ISimulationSettings.getInt(params, "max_episode_length");
        vmAllocationPolicy = ISimulationSettings.getStr(params, "vm_allocation_policy");
        algorithm = ISimulationSettings.getStr(params, "algorithm");
        sendObservationTreeArray = ISimulationSettings.getBool(params, "send_observation_tree_array");
        datacenters = (List<Map<String, Object>>) params.get("datacenters");
        vmTypes = (List<Map<String, Object>>) params.get("vm_types");
        hostsCount = computeHostsCount();
        datacenterCores = computeDatacenterCores();
    }

    public int getVmTypesCount() {
        return vmTypes.size();
    }

    public Map<String, Object> getVmTypeConfig(final int index) {
        return vmTypes.get(index);
    }

    public int getVmCoreCountByTypeIndex(final int index) {
        return ((Number) vmTypes.get(index).get("pes")).intValue();
    }

    public int getMinVmPes() {
        return ((Number) vmTypes.get(0).get("pes")).intValue();
    }

    public int getMaxVmPes() {
        return ((Number) vmTypes.get(vmTypes.size() - 1).get("pes")).intValue();
    }

    public long getDatacenterCores() {
        return datacenterCores;
    }

    @SuppressWarnings("unchecked")
    private int computeHostsCount() {
        int count = 0;
        for (Map<String, Object> dc : datacenters) {
            int dcAmount = ((Number) dc.getOrDefault("amount", 1)).intValue();
            List<Map<String, Object>> hosts = (List<Map<String, Object>>) dc.get("hosts");
            for (Map<String, Object> host : hosts) {
                count += dcAmount * ((Number) host.get("amount")).intValue();
            }
        }
        return count;
    }

    @SuppressWarnings("unchecked")
    private long computeDatacenterCores() {
        long cores = 0;
        for (Map<String, Object> dc : datacenters) {
            int dcAmount = ((Number) dc.getOrDefault("amount", 1)).intValue();
            List<Map<String, Object>> hosts = (List<Map<String, Object>>) dc.get("hosts");
            for (Map<String, Object> host : hosts) {
                int hostAmount = ((Number) host.get("amount")).intValue();
                int pes = ((Number) host.get("pes")).intValue();
                cores += (long) dcAmount * hostAmount * pes;
            }
        }
        return cores;
    }
}

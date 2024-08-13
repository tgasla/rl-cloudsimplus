package daislab.cspg;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.StatUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.Gson;

import static org.apache.commons.math3.stat.StatUtils.percentile;

public class WrappedSimulation {
    private static final Logger LOGGER =
        LoggerFactory.getLogger(WrappedSimulation.class.getSimpleName());
    // private static final int HISTORY_LENGTH = 30 * 60; // 30 * 60s = 1800s (30 minutes)
    
    private final List<CloudletDescriptor> initialJobsDescriptors;
    // private final List<String> metricsNames = Arrays.asList(
        // "hostCoresAllocatedToVmsRatio"
        // "avgCpuUtilization",
        // "p90CpuUtilization",
        // "avgMemoryUtilization",
        // "p90MemoryUtilization",
        // "waitingJobsRatioGlobal",
        // "waitingJobsRatioTimestep"
    // );

    private static final int minJobPes = 1;
    private static final int datacenterMetricsCount = 6;
    private static final int hostMetricsCount = 10;
    private static final int vmMetricsCount = 7;
    private static final int jobMetricsCount = 5;

    // private final MetricsStorage metricsStorage = new MetricsStorage(HISTORY_LENGTH, metricsNames);
    private final Gson gson = new Gson();
    private final String identifier;
    private final SimulationSettings settings;
    private final SimulationHistory simulationHistory;
    private final int hostsCount;
    private final int maxVmsCount;
    private final int maxJobsCount;
    private final int observationArrayRows;
    private final int observationArrayColumns;
    private CloudSimProxy cloudSimProxy;
    private int stepCount;
    private int episodeCount = 0;
    private double epJobWaitRewardMean = 0.0;
    private double epUtilRewardMean = 0.0;
    private int epValidCount = 0;
    private long epWaitingJobsCountMax = 0;
    private long epRunningVmsCountMax = 0;
    private double unutilizedActive;
    private double unutilizedAll;

    public WrappedSimulation(
        final String identifier,
        final SimulationSettings settings,
        final List<CloudletDescriptor> jobs
    ) {
        this.settings = settings;
        this.identifier = identifier;
        this.initialJobsDescriptors = jobs;
        this.simulationHistory = new SimulationHistory();
        this.hostsCount = settings.getDatacenterHostsCnt();
        this.maxVmsCount = settings.getDatacenterHostsCnt() * settings.getHostPeCnt() / settings.getBasicVmPeCnt();
        this.maxJobsCount = maxVmsCount * settings.getBasicVmPeCnt() / this.minJobPes;
        this.observationArrayRows = 1 + hostsCount + maxVmsCount + maxJobsCount;
        this.observationArrayColumns = Math.max(hostMetricsCount, Math.max(vmMetricsCount, jobMetricsCount));
        info("Creating simulation: " + identifier);
    }

    private void info(final String message) {
        LOGGER.info(getIdentifier() + " " + message);
    }

    private void debug(final String message) {
        LOGGER.debug(getIdentifier() + " " + message);
    }

    public String getIdentifier() {
        return identifier;
    }

    public SimulationResetResult reset() {
        info("Reset initiated");
        info("job count: " + initialJobsDescriptors.size());

        episodeCount++;
        resetStepCount();

        // metricsStorage.clear();

        List<Cloudlet> cloudlets = initialJobsDescriptors
            .stream()
            .map(CloudletDescriptor::toCloudlet)
            .collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(settings, cloudlets, episodeCount);

        double[][] obs = getObservation();
        resetEpisodeStats();
        SimulationStepInfo info = new SimulationStepInfo();

        return new SimulationResetResult(obs, info);
    }

    public void resetEpisodeStats() {
        resetEpJobWaitRewardMean();
        resetEpUtilRewardMean();
        resetEpValidCount();
        resetEpWaitingJobsCountMax();
        resetEpRunningVmsCountMax();
    }

    public void close() {
        info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.terminate();
        }
    }

    public String render() {
        // Map<String, double[]> renderedEnv = new HashMap<>();
        // for (int i = 0; i < metricsNames.size(); i++) {
        //     renderedEnv.put(
        //         metricsNames.get(i),
        //         metricsStorage.metricValuesAsPrimitives(metricsNames.get(i))
        //     );
        // }
        // return gson.toJson(renderedEnv);
        return new String();
    }

    public void validateSimulationReset() {
        if (cloudSimProxy == null) {
            throw new IllegalStateException("Simulation not reset! Please call the reset() function!");
        }
    }

    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();

        stepCount++;

        boolean isValid = executeAction(action);
        cloudSimProxy.runFor(settings.getTimestepInterval());
        // collectMetrics();

        boolean done = !cloudSimProxy.isRunning();
        double[][] observation = getObservation();
        double[] rewards = calculateReward(isValid);

        recordSimulationData(action, rewards);
        
        resetIfSimulationIsNotRunning();

        debug("Step finished (action: " + action[0] + ", " + action[1] + "), is done: " + done
            + " Length of future events queue: " + cloudSimProxy.getNumberOfFutureEvents());

        updateEpisodeStats(rewards[1], rewards[2], isValid);
        printEpisodeStatsDebug(rewards[1], rewards[2], rewards[3]);
        
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();

        SimulationStepInfo info = new SimulationStepInfo(
            rewards,
            getEpisodeRewardStats(),
            getCurrentTimestepMetrics(vmList),
            cloudSimProxy.getFinishedJobsWaitTimeLastInterval(), //jobWaitTime
            getUnutilizedStats(vmList)
        );

        return new SimulationStepResult(observation, rewards[0], done, info);
    }

    private List<double[][]> getCurrentTimestepMetrics(List<Vm> vmList) {
        List<double[][]> metrics = new ArrayList<>();
        
        metrics.add(collectHostMetrics());
        metrics.add(collectVmMetrics());
        metrics.add(collectJobMetrics());

        return metrics;
    }

    private double[] getUnutilizedStats(List<Vm> vmList) {
        double[] unutilizedStats = new double[2];

        unutilizedStats[0] = getUnutilizedVmCoresOverRunningVms(vmList);
        unutilizedStats[1] = getUnutilizedVmCoresOverAllHostCores(vmList);
        return unutilizedStats;
    }

    private List<Cloudlet> getCloudletList() {
        List<Cloudlet> cloudletList = cloudSimProxy.getBroker()
            .getVmExecList()
            .parallelStream()
            .map(Vm::getCloudletScheduler)
            .map(CloudletScheduler::getCloudletList)
            .flatMap(List::stream)
            .collect(Collectors.toList());
        return cloudletList;
    }

    private double getUnutilizedVmCoresOverRunningVms(List<Vm> vmList) {
        Long unutilizedVmCores = getUnutilizedVmCores(vmList);
        Long runningVmCores = getRunningVmCores(vmList);
        double unutilized = ((double) unutilizedVmCores / runningVmCores);
        return unutilized;
    }

    private double getUnutilizedVmCoresOverAllHostCores(List<Vm> vmList) {
        Long unutilizedVmCores = getUnutilizedVmCores(vmList);
        double unutilized = ((double) unutilizedVmCores / settings.getTotalHostCores());
        return unutilized;
    }

    private Long getUnutilizedVmCores(List<Vm> vmList) {
        Long unutilizedVmCores = vmList
            .parallelStream()
            .map(Vm::getFreePesNumber)
            .reduce(-1L, Long::sum);
        return unutilizedVmCores;
    }

    private Long getRunningVmsCount() {
        return cloudSimProxy.getBroker().getVmExecList().stream().count();
    }

    private Long getRunningCloudletsCount() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();

        Long runningCloudletCount = vmList
            .parallelStream()
            .map(Vm::getCloudletScheduler)
            .map(CloudletScheduler::getCloudletExecList)
            .mapToLong(List::size)
            .sum();
        return runningCloudletCount;
    }

    private Long getRunningVmCores(List<Vm> vmList) {
        Long runningVmCores = vmList
            .parallelStream()
            .map(Vm::getPesNumber)
            .reduce(1L, Long::sum);
        return runningVmCores;
    }

    private long vmCountByType(List<Vm> vmList, String type) {
        long filteredVmCount = vmList
        .stream()
        .filter(vm -> type.equals(vm.getDescription()))
        .count();

        return filteredVmCount;
    }

    private double[] collectDatacenterMetrics() {
        double[] datacenterMetrics = new double[] {
            // (double) cloudSimProxy.getAllocatedCores(),
            // (double) settings.getTotalHostCores(),
            getHostCoresAllocatedToVmsRatio(),
            // (double) settings.getDatacenterHostsCnt(),
            // (double) getRunningVmsCount(),
            // (double) getRunningCloudletsCount()
        };

        return datacenterMetrics;
    }

    private double[][] collectHostMetrics() {
        List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
        // We could also keep a Map<Host, Integer, Integer>
        //  hostId,  vmsRunning, pesUtilized
        double[][] hostMetrics = new double[hostList.size()][4];

        // int[] hostVmsRunningCount = new int[hostList.size()];
        // int[] hostPesUtilized = new int[hostList.size()];
        for (int i = 0; i < hostList.size(); i++) {
            Host host = hostList.get(i);
            List<Vm> vmList = hostList.get(i).getVmList();
            long smallVmCount = vmCountByType(vmList, "S");
            long mediumVmCount = vmCountByType(vmList, "M");
            long largeVmCount = vmCountByType(vmList, "L");

            hostMetrics[i] = new double[] {
                // host.getId(),
                // host.getVmList().size(),
                // smallVmCount,
                smallVmCount / (settings.getHostPeCnt() / cloudSimProxy.getVmCoreCountByType("S")),
                // mediumVmCount,
                mediumVmCount / (settings.getHostPeCnt() / cloudSimProxy.getVmCoreCountByType("M")),
                // largeVmCount,
                largeVmCount / (settings.getHostPeCnt() / cloudSimProxy.getVmCoreCountByType("L")),
                host.getBusyPesNumber() / host.getPesNumber()
            };
        }
        return hostMetrics;
    }

    private double[][] collectVmMetrics() {
        // consider adding cores utilized: vm.getPesNumber() - vm.getFreePesNumber()
        //  vmId,    vmPesNumber,  hostId,  jobsRunning
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        double[][] vmMetrics = new double[vmList.size()][1];
        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            vmMetrics[i] = new double[] {
                // vm.getId(),
                // vm.getHost().getId(),
                // vm.getCloudletScheduler().getCloudletList().size(),
                // vm.getCloudletScheduler().getCloudletExecList().size(),
                // vm.getCloudletScheduler().getCloudletWaitingList().size(),
                (vm.getPesNumber() - vm.getFreePesNumber()) / vm.getPesNumber()
            };
        }
        return vmMetrics;
    }

    private double[][] collectJobMetrics() {
        List<Cloudlet> cloudletList = getCloudletList();
        //   jobId,  jobPes,  vmId,    vmType,  hostId
        double[][] jobMetrics = new double[cloudletList.size()][1];
        for (int i = 0; i < cloudletList.size(); i++) {
            Cloudlet cloudlet = cloudletList.get(i);
            jobMetrics[i] = new double[] {
                // cloudlet.getId(),
                // cloudlet.getPesNumber(),
                // cloudlet.getVm().getId(),
                // cloudlet.getVm().getPesNumber(),
                // cloudlet.getVm().getHost().getId()
                cloudlet.getPesNumber() / cloudlet.getVm().getPesNumber()
            };
        }
        return jobMetrics;
    }

    // private double[] listOfArraysToArray(List<double[]> listOfArrays) {
    //     final int listSize = listOfArrays.size();
    //     final int arraySize = listOfArrays.get(0).length;
    //     // we know that each array in the list ahs the same size
    //     // so we calculate the final array size just by the size of the list
    //     // and the first array length
    //     double[] resultArray = new double[listSize * arraySize];

    //     int currentIndex = 0;
    //     for (double[] array : listOfArrays) {
    //         System.arraycopy(array, 0, resultArray, currentIndex, array.length);
    //         currentIndex += array.length;
    //     }
        
    //     return resultArray;
    // }

    private void updateValidCount(boolean isValid) {
        if (isValid) {
            epValidCount++;
        }
    }

    private void printEpisodeStatsDebug(
            double jobWaitReward,
            double utilReward,
            double invalidReward) {

        debug(" Mean episode job wait reward: " + getEpJobWaitRewardMean()
        + " Mean episode utilization reward: " + getEpUtilRewardMean()
        + " Max episode waiting jobs count: " + getEpWaitingJobsCountMax()
        + " Max episode running vms count: " + getEpRunningVmsCountMax()
        + " Job wait reward: " + jobWaitReward
        + " Utilization reward: " + utilReward
        + " Invalid reward: " + invalidReward);
    }

    private void updateEpisodeStats(double jobWaitReward, double utilReward, boolean isValid) {
        updateEpJobWaitRewardMean(-settings.getRewardJobWaitCoef() * jobWaitReward);
        updateEpUtilRewardMean(-settings.getRewardUtilizationCoef() * utilReward);
        updateValidCount(isValid);

        updateEpWaitingJobsCountMax(cloudSimProxy.getWaitingJobsCount());
        updateEpRunningVmsCountMax(cloudSimProxy.getBroker().getVmExecList().size());   
    }

    private List<Object> getEpisodeRewardStats() {
        List<Object> stats = new ArrayList<>();

        stats.add(getEpJobWaitRewardMean());
        stats.add(getEpUtilRewardMean());
        stats.add(getEpValidCount());

        return stats;
    }

    private long continuousToDiscrete(final double continuous, final long bucketsNum) {
    /*
        * Explanation:
        * floor(continuous * bucketsNum) will give you the discrete value
        * but, in case of cont = 1, then the discrete value 
        * will be equal to bucketsNum.
        * However, we want to map the continuous value to the
        * range of [0,bucketsNum-1].
        * So, Math.min ensures that the maximum allowed
        * discrete value will be bucketsNum-1.
    */
        final long discrete = (long) Math.min(Math.floor(continuous * bucketsNum), bucketsNum - 1);
        return discrete;
    }

    private void resetIfSimulationIsNotRunning() {
        if (!cloudSimProxy.isRunning()) {
            simulationHistory.logHistory();
            simulationHistory.reset();
        }
    }

    private void recordSimulationData(int[] action, double[] reward) {
        simulationHistory.record("action[0]", action[0]);
        simulationHistory.record("action[1]", action[1]);
        simulationHistory.record("action[2]", action[2]);
        simulationHistory.record("totalReward", reward[0]);
        simulationHistory.record("jobWaitReward", reward[1]);
        simulationHistory.record("utilReward", reward[2]);
        simulationHistory.record("invalidReward", reward[3]);
        simulationHistory.record("totalCost", cloudSimProxy.getRunningCost());
        simulationHistory.record("vmExecCount", cloudSimProxy.getBroker().getVmExecList().size());
    }

    private boolean executeAction(final int[] action) {

        final boolean isValid;

        debug("action is " + action[0] + ", " + action[1] + ", " + action[2]);

        // [action, id, type^]
        // action = {0: do nothing, 1: create vm, 2: destroy vm}
        // type = {0: small, 1: medium, 2: large}
        // ^ needed only when action = 1

        if (action[0] == 1) {
            final int hostId = action[1];
            final int vmTypeIndex = action[2];
            isValid = addNewVm(CloudSimProxy.VM_TYPES[vmTypeIndex], hostId);
            return isValid;
        }

        else if (action[0] == 2) {
            final int vmIndex = action[1];
            isValid = removeVm(vmIndex);
            return isValid;
        }

        else {
            return true;
        }
    }

        // final long id;
        // final int index;
        // final int vmTypeIndex;

        // // action < 0 destroys the VM with VM.index = index
        // if (action[0] < 0) {
        //     index = (int) continuousToDiscrete(
        //         Math.abs(action[0]),
        //         cloudSimProxy.getBroker().getVmExecList().size());

        //     if (index < 0) {
        //         debug("No active Vms. Ignoring action...");
        //         return false;
        //     }
        //     debug("translated action[0] = " + index);
        //     debug("will try to destroy vm with index = " + index);
        //     isValid = removeVm(index);
        //     return isValid;
        // }

        // // action > 0 creates a VM in host host.id = id
        // // and Vm.type = action[1]
        // else if (action[0] > 0) {
        //     id = continuousToDiscrete(
        //         action[0],
        //         settings.getDatacenterHostsCnt());

        //     vmTypeIndex = (int) continuousToDiscrete(
        //         action[1],
        //         CloudSimProxy.VM_TYPES.length);

        //     debug("Translated action[0] = " + id);
        //     debug("Will try to create a new vm at host with id = " 
        //             + id + " of type " + CloudSimProxy.VM_TYPES[vmTypeIndex]);
        //     isValid = addNewVm(CloudSimProxy.VM_TYPES[vmTypeIndex], id);
        //     return isValid;
        // }
        // else {
        //     // action[0] = 0 does nothing
        //     return true;
        // }

    private boolean removeVm(final int index) {
        if (!cloudSimProxy.removeVm(index)) {
            debug("Removing a VM with index " + index + " action is invalid. Ignoring.");
            return false;
        }
        return true;
    }

    // adds a new vm to the host with hostid if possible
    private boolean addNewVm(final String type, final long hostId) {
        if (!cloudSimProxy.addNewVm(type, hostId)) {
            debug("Adding a VM of type " + type + " to host " + hostId + " is invalid. Ignoring");
            return false;
        }
        return true;
    }

    private double percentileOrZero(final double[] values, final double percentile) {
        return values.length > 0 ? percentile(values, percentile) : 0;
    }

    // private void collectMetrics() {
        // double[] cpuPercentUsage = cloudSimProxy.getVmCpuUsage();
        // Arrays.sort(cpuPercentUsage);

        // double[] memPercentageUsage = cloudSimProxy.getVmMemoryUsage();
        // Arrays.sort(memPercentageUsage);

        // double waitingJobsRatioGlobal = getWaitingJobsRatioGlobal();
        // double waitingJobsRatioTimestep = getWaitingJobsRatioTimestep();

        // metricsStorage.updateMetric(
            // "hostCoresAllocatedToVmsRatio",
            // getHostCoresAllocatedToVmsRatio());
        // metricsStorage.updateMetric(
        //     "avgCpuUtilization",
        //     safeMean(cpuPercentUsage));
        // metricsStorage.updateMetric(
        //     "p90CpuUtilization", 
        //     percentileOrZero(cpuPercentUsage, 0.90));
        // metricsStorage.updateMetric(
        //     "avgMemoryUtilization",
        //     safeMean(memPercentageUsage));
        // metricsStorage.updateMetric(
        // "p90MemoryUtilization", 
        // percentileOrZero(memPercentageUsage, 0.90));
        // metricsStorage.updateMetric(
        //     "waitingJobsRatioGlobal",
        //     waitingJobsRatioGlobal);
        // metricsStorage.updateMetric(
        //     "waitingJobsRatioTimestep",
        //     waitingJobsRatioTimestep);
    // }

    private double getWaitingJobsRatioTimestep() {
        final int submittedJobsCountLastInterval =
            cloudSimProxy.getSubmittedJobsCountLastInterval();

        return submittedJobsCountLastInterval > 0
            ? cloudSimProxy.getWaitingJobsCountLastInterval() 
            / (double) submittedJobsCountLastInterval : 0.0;
    }

    private double getWaitingJobsRatioGlobal() {
        final int submittedJobsCount = cloudSimProxy.getSubmittedJobsCount();

        return submittedJobsCount > 0 
            ? cloudSimProxy.getWaitingJobsCount() / (double) submittedJobsCount : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        return ((double) cloudSimProxy.getAllocatedCores()) / settings.getTotalHostCores();
    }

    // private double[] getObservation() {
        // return new double[] {
            // metricsStorage.getLastMetricValue("hostCoresAllocatedToVmsRatio"),
            // metricsStorage.getLastMetricValue("avgCpuUtilization"),
            // metricsStorage.getLastMetricValue("p90CpuUtilization"),
            // metricsStorage.getLastMetricValue("avgMemoryUtilization"),
            // metricsStorage.getLastMetricValue("p90MemoryUtilization"),
            // metricsStorage.getLastMetricValue("waitingJobsRatioGlobal"),
            // metricsStorage.getLastMetricValue("waitingJobsRatioTimestep")
        // };
    // }

    private void print2dArray(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                System.out.print(array[i][j] + " ");
            }
            System.out.print("\n");
        }
    }

    private double[][] getObservation() {
        final double[] datacenterMetrics = collectDatacenterMetrics();
        final double[][] hostMetrics = collectHostMetrics();
        final double[][] vmMetrics = collectVmMetrics();
        final double[][] jobMetrics = collectJobMetrics();

        double[][] observation = new double[observationArrayRows][observationArrayColumns];
        int currentRow = 0;

        for (int j = 0; j < datacenterMetrics.length; j++) {
            observation[currentRow][j] = datacenterMetrics[j];
        }
        currentRow++;

        for (int i = 0; i < hostMetrics.length; i++) {
            for (int j = 0; j < hostMetrics[0].length; j++) {
                observation[currentRow][j] = hostMetrics[i][j];
            }
            currentRow++;
        }

        for (int i = 0; i < vmMetrics.length; i++) {
            for (int j = 0; j < vmMetrics[0].length; j++) {
                observation[currentRow][j] = vmMetrics[i][j];
            }
            currentRow++;
        }

        currentRow = 1 + this.hostsCount + this.maxVmsCount;

        for (int i = 0; i < jobMetrics.length; i++) {
            for (int j = 0; j < jobMetrics[0].length; j++) {
                observation[currentRow][j] = jobMetrics[i][j];
            }
            currentRow++;
        }

        // print2dArray(observation);
        
        return observation;
    }

    private double safeMean(final double[] values) {
        return values.length > 0 ? StatUtils.mean(values) : 0.0;
    }

    private double[] calculateReward(final boolean isValid) {
        double[] rewards = new double[4];
        final int rewardMultiplier = 1;
        /* reward is the negative cost of running the infrastructure
         * minus any penalties from jobs waiting in the queue
         * minus penalty if action was invalid
        */
        final double jobWaitCoef = settings.getRewardJobWaitCoef();
        final double utilizationCoef = settings.getRewardUtilizationCoef();
        final double invalidCoef = settings.getRewardInvalidCoef();
        
        final double jobWaitReward = - jobWaitCoef * getWaitingJobsRatioGlobal();
        final double utilReward = - utilizationCoef * getHostCoresAllocatedToVmsRatio();
        final double invalidReward = - invalidCoef * (isValid ? 0 : 1);
        
        double totalReward = jobWaitReward + utilReward + invalidReward;

        totalReward *= rewardMultiplier;

        if (!isValid) {
            debug("Penalty given to the agent because the selected action was not possible");
        }

        rewards[0] = totalReward;
        rewards[1] = jobWaitReward;
        rewards[2] = utilReward;
        rewards[3] = invalidReward;
        return rewards;
    }

    public int getStepCount() {
        return stepCount;
    }

    private void resetStepCount() {
        stepCount = 0;
    }

    private void resetEpRunningVmsCountMax() {
        epRunningVmsCountMax = 0;
    }

    private void resetEpWaitingJobsCountMax() {
        epWaitingJobsCountMax = 0;
    }

    private long getEpRunningVmsCountMax() {
        return epRunningVmsCountMax;
    }

    private long getEpWaitingJobsCountMax() {
        return epWaitingJobsCountMax;
    }

    private void updateEpWaitingJobsCountMax(long waitingJobsCount) {
        if (waitingJobsCount > epWaitingJobsCountMax) {
            epWaitingJobsCountMax = waitingJobsCount;
        }
    }

    private void updateEpRunningVmsCountMax(long runningVms) {
        if (runningVms > epRunningVmsCountMax) {
            epRunningVmsCountMax = runningVms;
        }
    }
    
    private void updateEpJobWaitRewardMean(double jobWaitReward) {
        epJobWaitRewardMean = (epJobWaitRewardMean * (stepCount - 1) + jobWaitReward) / stepCount;
    }

    private void updateEpUtilRewardMean(double utilReward) {
        epUtilRewardMean = (epUtilRewardMean * (stepCount - 1) +  utilReward) / stepCount;
    }

    private double getEpJobWaitRewardMean() {
        return epJobWaitRewardMean;
    }

    private double getEpUtilRewardMean() {
        return epUtilRewardMean;
    }

    private int getEpValidCount() {
        return epValidCount;
    }

    private void resetEpJobWaitRewardMean() {
        epJobWaitRewardMean = 0.0;
    }
    
    private void resetEpUtilRewardMean() {
        epUtilRewardMean = 0.0;
    }

    private void resetEpValidCount() {
        epValidCount = 0;
    }

    public void seed() {
        // there is no randomness so far...
    }

    public double clock() {
        return cloudSimProxy.clock();
    }

    public SimulationSettings getSimulationSettings() {
        return settings;
    }

    public void printJobStats() {
        cloudSimProxy.printJobStats();
    }
}

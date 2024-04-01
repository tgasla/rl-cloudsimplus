package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.StatUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.Gson;

import static org.apache.commons.math3.stat.StatUtils.percentile;

public class WrappedSimulation {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(WrappedSimulation.class.getSimpleName());
    private static final int HISTORY_LENGTH = 30 * 60; // 30 * 60s = 1800s (30 minutes)
    
    private final double queueWaitPenalty;
    private final List<CloudletDescriptor> initialJobsDescriptors;
    private final double simulationSpeedUp;
    private final List<String> metricsNames = Arrays.asList(
            "vmAllocatedRatioHistory",
            "avgCPUUtilizationHistory",
            "p90CPUUtilizationHistory",
            "avgMemoryUtilizationHistory",
            "p90MemoryUtilizationHistory",
            "waitingJobsRatioGlobalHistory",
            "waitingJobsRatioRecentHistory"
    );

    private final MetricsStorage metricsStorage = new MetricsStorage(HISTORY_LENGTH, metricsNames);
    private final Gson gson = new Gson();
    private final double INTERVAL = 1.0;
    private final String identifier;
    private final Map<String, Integer> initialVmsCount;
    private final SimulationSettings settings;
    private final SimulationHistory simulationHistory;
    private CloudSimProxy cloudSimProxy;
    private VmCounter vmCounter;
    private double maxCost = 0.0;
    private int validCount = 0;
    private int actionCount = 0;

    public WrappedSimulation(final SimulationSettings simulationSettings,
                             final String identifier,
                             final Map<String, Integer> initialVmsCount,
                             final double simulationSpeedUp,
                             final double queueWaitPenalty,
                             final List<CloudletDescriptor> jobs) {
        this.settings = simulationSettings;
        this.identifier = identifier;
        this.initialVmsCount = initialVmsCount;
        this.initialJobsDescriptors = jobs;
        this.simulationSpeedUp = simulationSpeedUp;
        this.queueWaitPenalty = queueWaitPenalty;
        this.simulationHistory = new SimulationHistory();

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

        // first attempt to store some memory
        metricsStorage.clear();
        vmCounter = new VmCounter(settings.getMaxVmsPerSize());
        vmCounter.initializeCapacity(
                CloudSimProxy.SMALL, initialVmsCount.get(CloudSimProxy.SMALL));
        vmCounter.initializeCapacity(
                CloudSimProxy.MEDIUM, initialVmsCount.get(CloudSimProxy.MEDIUM));
        vmCounter.initializeCapacity(
                CloudSimProxy.LARGE, initialVmsCount.get(CloudSimProxy.LARGE));

        List<Cloudlet> cloudlets = initialJobsDescriptors
                .stream()
                .map(CloudletDescriptor::toCloudlet)
                .collect(Collectors.toList());
        debug("Calling CloudSimProxy object...");
        cloudSimProxy = new CloudSimProxy(
                settings, 
                initialVmsCount,
                cloudlets, 
                simulationSpeedUp);

        double[] obs = getObservation();

        resetMaxCost();
        resetValidCount();
        resetActionCount();

        SimulationStepInfo info = new SimulationStepInfo(
                validCount, 
                actionCount,
                getMaxCost());

        return new SimulationResetResult(obs, info);
    }

    public void close() {
        info("Simulation is synchronous - doing nothing");
    }

    public String render() {
        double[][] renderedEnv = {
                metricsStorage.metricValuesAsPrimitives("vmAllocatedRatioHistory"),
                metricsStorage.metricValuesAsPrimitives("avgCPUUtilizationHistory"),
                metricsStorage.metricValuesAsPrimitives("p90CPUUtilizationHistory"),
                metricsStorage.metricValuesAsPrimitives("avgMemoryUtilizationHistory"),
                metricsStorage.metricValuesAsPrimitives("p90MemoryUtilizationHistory"),
                metricsStorage.metricValuesAsPrimitives("waitingJobsRatioGlobalHistory"),
                metricsStorage.metricValuesAsPrimitives("waitingJobsRatioRecentHistory")
        };

        return gson.toJson(renderedEnv);
    }

    public SimulationStepResult step(final double[] action) {

        if (cloudSimProxy == null) {
            throw new RuntimeException("Simulation not reset! Please call the reset() function!");
        }

        debug("Executing action: " + action[0] + ", " + action[1]);

        long startAction = System.nanoTime();
        boolean isValid = executeAction(action);
        long stopAction = System.nanoTime();
        cloudSimProxy.runFor(INTERVAL);

        long startMetrics = System.nanoTime();
        collectMetrics();
        long stopMetrics = System.nanoTime();

        boolean done = !cloudSimProxy.isRunning();
        double[] observation = getObservation();
        double reward = calculateReward(isValid);

        simulationHistory.record("action[0]", action[0]);
        simulationHistory.record("action[1]", action[1]);
        simulationHistory.record("reward", reward);
        simulationHistory.record("resourceCost", cloudSimProxy.getRunningCost());
        simulationHistory.record(
                "small_vms", vmCounter.getStartedVms(CloudSimProxy.SMALL));
        simulationHistory.record(
                "medium_vms", vmCounter.getStartedVms(CloudSimProxy.MEDIUM));
        simulationHistory.record(
                "large_vms", vmCounter.getStartedVms(CloudSimProxy.LARGE));

        if (!cloudSimProxy.isRunning()) {
            simulationHistory.logHistory();
            simulationHistory.reset();
        }

        double metricsTime = (stopMetrics - startMetrics) / 1_000_000_000d;
        double actionTime = (stopAction - startAction) / 1_000_000_000d;
        debug("Step finished (action: " + action[0] + ", " + action[1] + ") is done: " + done
                + " Length of future events queue: " + cloudSimProxy.getNumberOfFutureEvents()
                + " Metrics (s): " + metricsTime
                + " Action (s): " + actionTime);

        double cost = cloudSimProxy.getRunningCost();
        updateMaxCost(cost);

        debug("Max cost is: " + maxCost);

        if (isValid) {
            validCount += 1;
        }

        actionCount += 1;

        SimulationStepInfo info = new SimulationStepInfo(
                validCount,
                actionCount,
                getMaxCost());

        return new SimulationStepResult(
                observation,
                reward,
                done,
                info
        );
    }

    private long continuousToPositiveDiscrete(
            final double continuous, 
            final long maxDiscreteValue) {
        final long discrete = Long.valueOf(Math.round(continuous * maxDiscreteValue));
        return Math.abs(discrete);
    }

    private boolean executeAction(final double[] action) {

        debug("action is " + action[0] + ", " + action[1]);

        boolean isValid = true;
        final long id;
        final int vmTypeIndex;

        // action[0] = 0 does nothing

        // action < 0 destroys a VM with Vm.id = id
        if (action[0] < 0) {
            id = continuousToPositiveDiscrete(
                action[0],
                cloudSimProxy.getLastCreatedVmId());
            debug("translated action[0] = " + id);
            debug("will try to destroy vm with id = " + id);
            isValid = removeVm(id);
        }
        // action > 0 creates a VM in host host.id = id
        // and Vm.type = action[1]
        else if (action[0] > 0) {
            id = continuousToPositiveDiscrete(
                action[0],
                settings.getDatacenterHostsCnt() - 1);

            vmTypeIndex = (int) continuousToPositiveDiscrete(
                action[1],
                CloudSimProxy.VM_TYPES.length - 1);

            debug("translated action[0] = " + id);
            debug("will try to create a new Vm on the same host as the vm with id = " 
                    + id + " of type " + CloudSimProxy.VM_TYPES[vmTypeIndex]);
            isValid = addNewVm(CloudSimProxy.VM_TYPES[vmTypeIndex], id);
        }
        return isValid;
    }

    private boolean removeVm(final long id) {
        String vmToKillType = cloudSimProxy.removeVm(id);
        if (vmToKillType == null) {
            debug("Remove vm with id " + id + " request was ignored.");
            return false;
        }
        vmCounter.recordRemovedVm(vmToKillType);
        return true;
    }

    private boolean removeRandomVm(final String type) {
        if (!cloudSimProxy.removeRandomVm(type)) {
            debug("Removing a VM of type "
                    + type + " requested but the request was ignored. Stats: "
                    + " S: " + vmCounter.getStartedVms(CloudSimProxy.SMALL)
                    + " M: " + vmCounter.getStartedVms(CloudSimProxy.MEDIUM)
                    + " L: " + vmCounter.getStartedVms(CloudSimProxy.LARGE)
            );
            return false;
        }
        vmCounter.recordRemovedVm(type);
        return true;
    }

    // adds a new vm to the same host as the vm with vmId if possible
    private boolean addNewVm(final String type, final long hostId) {
        // TODO: this vmCounter class should be changed so that it tracks
        // all the vms that each host has.
        // Not 3 numbers across the whole datacenter but instead,
        // 3 numbers per host Map<<int>, <Map<String, int>>>
        //                       <hostId, <"VM_SIZE", vmCount>>
        if (!vmCounter.hasCapacity(type)) {
            debug("Adding a VM of type "
                    + type + " requested but the request was ignored (MAX_VMS_PER_SIZE "
                    + settings.getMaxVmsPerSize() + " reached) Stats: "
                    + " S: " + vmCounter.getStartedVms(CloudSimProxy.SMALL)
                    + " M: " + vmCounter.getStartedVms(CloudSimProxy.MEDIUM)
                    + " L: " + vmCounter.getStartedVms(CloudSimProxy.LARGE)
            );
            return false;
        }

        if (!cloudSimProxy.addNewVm(type, hostId)) {
            debug("Adding a VM of type " + type + " to host "
                    + "was requested but the request was ignored "
                    + "because the host is not suitable");
            return false;
        }

        vmCounter.recordNewVm(type);
        return true;
    }

    private boolean addNewVm(final String type) {
        if (vmCounter.hasCapacity(type)) {
            cloudSimProxy.addNewVm(type);
            vmCounter.recordNewVm(type);
            return true;
        }
        else {
            debug("Adding a VM of type "
                    + type + " requested but the request was ignored (MAX_VMS_PER_SIZE "
                    + settings.getMaxVmsPerSize() + " reached) Stats: "
                    + " S: " + vmCounter.getStartedVms(CloudSimProxy.SMALL)
                    + " M: " + vmCounter.getStartedVms(CloudSimProxy.MEDIUM)
                    + " L: " + vmCounter.getStartedVms(CloudSimProxy.LARGE)
            );
            return false;
        }
    }

    private double percentileOrZero(
            final double[] values, 
            final double percentile) {

        if (values.length == 0) {
            return 0;
        }

        return percentile(values, percentile);
    }

    private void collectMetrics() {
        double[] cpuPercentUsage = cloudSimProxy.getVmCpuUsage();
        Arrays.sort(cpuPercentUsage);

        double[] memPercentageUsage = cloudSimProxy.getVmMemoryUsage();
        Arrays.sort(memPercentageUsage);

        double waitingJobsRatioGlobal = getWaitingJobsRatioGlobal();
        double waitingJobsRatioRecent = getWaitingJobsRatioRecent();

        metricsStorage.updateMetric("vmAllocatedRatioHistory", getVmAllocatedRatio());
        metricsStorage.updateMetric("avgCPUUtilizationHistory", safeMean(cpuPercentUsage));
        metricsStorage.updateMetric(
                "p90CPUUtilizationHistory", 
                percentileOrZero(cpuPercentUsage, 0.90));
        metricsStorage.updateMetric("avgMemoryUtilizationHistory", safeMean(memPercentageUsage));
        metricsStorage.updateMetric(
                "p90MemoryUtilizationHistory", 
                percentileOrZero(memPercentageUsage, 0.90));
        metricsStorage.updateMetric("waitingJobsRatioGlobalHistory", waitingJobsRatioGlobal);
        metricsStorage.updateMetric("waitingJobsRatioRecentHistory", waitingJobsRatioRecent);
    }

    private double getWaitingJobsRatioRecent() {
        final int submittedJobsCountLastInterval =
            cloudSimProxy.getSubmittedJobsCountLastInterval();
        if (submittedJobsCountLastInterval == 0) {
            return 0.0;
        }
        return cloudSimProxy.getWaitingJobsCountInterval(INTERVAL) 
                / (double) submittedJobsCountLastInterval;
    }

    private double getWaitingJobsRatioGlobal() {
        final int submittedJobsCount = cloudSimProxy.getSubmittedJobsCount();
        if (submittedJobsCount == 0) {
            return 0.0;
        }

        return cloudSimProxy.getWaitingJobsCount() / (double) submittedJobsCount;
    }

    private double getVmAllocatedRatio() {
        return ((double) cloudSimProxy.getNumberOfActiveCores()) / settings.getAvailableCores();
    }


    private double[] getObservation() {
        return new double[] {
                metricsStorage.getLastMetricValue("vmAllocatedRatioHistory"),
                metricsStorage.getLastMetricValue("avgCPUUtilizationHistory"),
                metricsStorage.getLastMetricValue("p90CPUUtilizationHistory"),
                metricsStorage.getLastMetricValue("avgMemoryUtilizationHistory"),
                metricsStorage.getLastMetricValue("p90MemoryUtilizationHistory"),
                metricsStorage.getLastMetricValue("waitingJobsRatioGlobalHistory"),
                metricsStorage.getLastMetricValue("waitingJobsRatioRecentHistory")
        };
    }

    private double safeMean(final double[] cpuPercentUsage) {
        if (cpuPercentUsage.length == 0) {
            return 0.0;
        }

        if (cpuPercentUsage.length == 1) {
            return cpuPercentUsage[0];
        }

        return StatUtils.mean(cpuPercentUsage);
    }

    private double calculateReward(final boolean isValid) {
        final int penalty = (isValid) ? 0 : 1000;
        final double vmCostCoef = 1;
        final double waitingJobsCoef = 1;

        if (!isValid) {
            info("Penalty given to the agent because the action was not possible");
        }
        // reward is the negative cost of running the infrastructure
        // - any penalties from jobs waiting in the queue
        // - penalty if action was invalid
        final double vmRunningCostTerm = cloudSimProxy.getRunningCost();
        final double waitingJobsTerm =
                cloudSimProxy.getWaitingJobsCount() 
                * queueWaitPenalty * simulationSpeedUp;
                
        return - vmCostCoef * vmRunningCostTerm 
                - waitingJobsCoef * waitingJobsTerm 
                - penalty;
    }

    private void updateMaxCost(double cost) {
        if (cost > maxCost) {
            maxCost = cost;
        }
    }

    private void resetMaxCost() {
        maxCost = 0.0;
    }

    public double getMaxCost() {
        return maxCost;
    }

    private void resetActionCount() {
        actionCount = 0;
    }

    private void resetValidCount() {
        validCount = 0;
    }

    public void seed() {
        // there is no randomness so far...
    }

    public double clock() {
        return cloudSimProxy.clock();
    }

    public CloudSimProxy getSimulation() {
        return cloudSimProxy;
    }

    public SimulationSettings getSimulationSettings() {
        return settings;
    }
}

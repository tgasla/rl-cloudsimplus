package daislab.cspg;

import com.google.gson.Gson;
import org.apache.commons.math3.stat.StatUtils;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

    public double[] reset() {
        info("Reset initiated");

        // first attempt to store some memory
        metricsStorage.clear();
        this.vmCounter = new VmCounter(this.settings.getMaxVmsPerSize());
        this.vmCounter.initializeCapacity(
                CloudSimProxy.SMALL, initialVmsCount.get(CloudSimProxy.SMALL));
        this.vmCounter.initializeCapacity(
                CloudSimProxy.MEDIUM, initialVmsCount.get(CloudSimProxy.MEDIUM));
        this.vmCounter.initializeCapacity(
                CloudSimProxy.LARGE, initialVmsCount.get(CloudSimProxy.LARGE));

        List<Cloudlet> cloudlets = initialJobsDescriptors
                .stream()
                .map(CloudletDescriptor::toCloudlet)
                .collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(
                settings, 
                initialVmsCount, 
                cloudlets, 
                simulationSpeedUp);

        double[] obs = getObservation();
        return obs;
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

    public SimulationStepResult step(final int action) {

        if (cloudSimProxy == null) {
            throw new RuntimeException("Simulation not reset! Please call the reset() function!");
        }

        debug("Executing action: " + action);

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

        this.simulationHistory.record("action", action);
        this.simulationHistory.record("reward", reward);
        this.simulationHistory.record("resourceCost", cloudSimProxy.getRunningCost());
        this.simulationHistory.record(
                "small_vms", this.vmCounter.getStartedVms(CloudSimProxy.SMALL));
        this.simulationHistory.record(
                "medium_vms", this.vmCounter.getStartedVms(CloudSimProxy.MEDIUM));
        this.simulationHistory.record(
                "large_vms", this.vmCounter.getStartedVms(CloudSimProxy.LARGE));

        if (!cloudSimProxy.isRunning()) {
            this.simulationHistory.logHistory();
            this.simulationHistory.reset();
        }

        double metricsTime = (stopMetrics - startMetrics) / 1_000_000_000d;
        double actionTime = (stopAction - startAction) / 1_000_000_000d;
        debug("Step finished (action: " + action + ") is done: " + done
                + " Length of future events queue: " + cloudSimProxy.getNumberOfFutureEvents()
                + " Metrics (s): " + metricsTime
                + " Action (s): " + actionTime);

        return new SimulationStepResult(
                done,
                observation,
                reward
        );
    }

    private boolean executeAction(final int action) {
        boolean isValid = true;

        switch (action) {
            case 0:
                // nothing happens
                break;
            case 1:
                // adding a new vm
                isValid = addNewVm(CloudSimProxy.SMALL);
                break;
            case 2:
                // removing randomly one of the vms
                isValid = removeVM(CloudSimProxy.SMALL);
                break;
            case 3:
                isValid = addNewVm(CloudSimProxy.MEDIUM);
                break;
            case 4:
                isValid = removeVM(CloudSimProxy.MEDIUM);
                break;
            case 5:
                isValid = addNewVm(CloudSimProxy.LARGE);
                break;
            case 6:
                isValid = removeVM(CloudSimProxy.LARGE);
                break;
        }
        return isValid;
    }

    private boolean removeVM(final String type) {
        if (cloudSimProxy.removeRandomVm(type)) {
            this.vmCounter.recordRemovedVm(type);
            return true;
        } else {
            debug("Removing a VM of type "
                    + type + " requested but the request was ignored. Stats: "
                    + " S: " + this.vmCounter.getStartedVms(CloudSimProxy.SMALL)
                    + " M: " + this.vmCounter.getStartedVms(CloudSimProxy.MEDIUM)
                    + " L: " + this.vmCounter.getStartedVms(CloudSimProxy.LARGE)
            );
            return false;
        }
    }

    private boolean addNewVm(final String type) {
        if (vmCounter.hasCapacity(type)) {
            cloudSimProxy.addNewVm(type);
            vmCounter.recordNewVm(type);
            return true;
        } else {
            debug("Adding a VM of type "
                    + type
                    + " requested but the request was ignored (MAX_VMS_PER_SIZE "
                    + this.settings.getMaxVmsPerSize() + " reached) Stats: "
                    + " S: " + this.vmCounter.getStartedVms(CloudSimProxy.SMALL)
                    + " M: " + this.vmCounter.getStartedVms(CloudSimProxy.MEDIUM)
                    + " L: " + this.vmCounter.getStartedVms(CloudSimProxy.LARGE)
            );
            return false;
        }
    }

    private double percentilegetEnvironmentVariable(
            final double[] values, 
            final double percentile, 
            final double defaultValue) {

        if (values.length == 0) {
            return defaultValue;
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
                percentilegetEnvironmentVariable(cpuPercentUsage, 0.90, 0));
        metricsStorage.updateMetric("avgMemoryUtilizationHistory", safeMean(memPercentageUsage));
        metricsStorage.updateMetric(
                "p90MemoryUtilizationHistory", 
                percentilegetEnvironmentVariable(memPercentageUsage, 0.90, 0));
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
        final int penaltyMultiplier = (isValid) ? 1 : 1000;
        if (!isValid) {
            LOGGER.info("Penalty given to agent because action was not possible");
        }
        // reward is the negative cost of running the infrastructure
        // - any penalties from jobs waiting in the queue
        final double vmRunningCost = cloudSimProxy.getRunningCost();
        final double penalty =
                  this.cloudSimProxy.getWaitingJobsCount() 
                * this.queueWaitPenalty 
                * simulationSpeedUp;
        return - penaltyMultiplier * (vmRunningCost + penalty);
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
        return this.settings;
    }
}

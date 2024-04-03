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
    
    private final List<CloudletDescriptor> initialJobsDescriptors;
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
    private int validCount = 0;
    private double maxCost = 0.0;
    private double meanJobWaitPenalty = 0.0;
    private double meanCostPenalty = 0.0;
    private int stepCount;

    public WrappedSimulation(final SimulationSettings simulationSettings,
                             final String identifier,
                             final Map<String, Integer> initialVmsCount,
                             final List<CloudletDescriptor> jobs) {
        this.settings = simulationSettings;
        this.identifier = identifier;
        this.initialVmsCount = initialVmsCount;
        this.initialJobsDescriptors = jobs;
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

        stepCount = 0;

        // first attempt to store some memory
        metricsStorage.clear();

        List<Cloudlet> cloudlets = initialJobsDescriptors
                .stream()
                .map(CloudletDescriptor::toCloudlet)
                .collect(Collectors.toList());
        debug("Calling CloudSimProxy object...");
        cloudSimProxy = new CloudSimProxy(
                settings, 
                initialVmsCount,
                cloudlets);

        double[] obs = getObservation();

        resetMaxCost();
        resetValidCount();
        resetMeanJobWaitPenalty();
        resetMeanCostPenalty();

        SimulationStepInfo info = new SimulationStepInfo(0,0,0);

        return new SimulationResetResult(obs, info);
    }

    public void close() {
        info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.getSimulation().terminate();
        }
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

        stepCount++;

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
                "vmExecCount", cloudSimProxy.getBroker().getVmExecList().size());

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
        double jobWaitPenalty =
                cloudSimProxy.getWaitingJobsCount() 
                * settings.getQueueWaitPenalty() * settings.getSimulationSpeedup();

        updateMaxCost(cost);
        updateMeanJobWaitPenalty(-jobWaitPenalty);
        updateMeanCostPenalty(-cost);

        debug("Max cost: " + getMaxCost() + "\n"
                + "Mean job wait penalty: " + getMeanJobWaitPenalty() + "\n"
                + "Mean cost penalty: " + getMeanCostPenalty() + "\n");

        if (isValid) {
            validCount++;
        }
        
        SimulationStepInfo info = new SimulationStepInfo(
                getValidCount(),
                getMeanJobWaitPenalty(),
                getMeanCostPenalty());

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

        final boolean isValid;
        final long id;
        final int vmTypeIndex;

        // action < 0 destroys a VM with Vm.id = id
        if (action[0] < 0) {
            id = continuousToPositiveDiscrete(
                action[0],
                cloudSimProxy.getLastCreatedVmId());
            debug("translated action[0] = " + id);
            debug("will try to destroy vm with id = " + id);
            isValid = removeVm(id);
            return isValid;
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

            debug("Translated action[0] = " + id);
            debug("Will try to create a new Vm on the same host as the vm with id = " 
                    + id + " of type " + CloudSimProxy.VM_TYPES[vmTypeIndex]);
            isValid = addNewVm(CloudSimProxy.VM_TYPES[vmTypeIndex], id);
            return isValid;
        }
        else {
            // action[0] = 0 does nothing
            return true;
        }
    }

    private boolean removeVm(final long id) {
        if (!cloudSimProxy.removeVm(id)) {
            debug("Removing a VM with id " + id + " action is invalid. Ignoring.");
            return false;
        }
        return true;
    }

    // private boolean removeRandomVm(final String type) {
    //     if (!cloudSimProxy.removeRandomVm(type)) {
    //         debug("Removing a random VM of type " + type + " is invalid. Ignoring.");
    //         return false;
    //     }
    //     return true;
    // }

    // adds a new vm to the host with hostid if possible
    private boolean addNewVm(final String type, final long hostId) {
        if (!cloudSimProxy.addNewVm(type, hostId)) {
            debug("Adding a VM of type " + type + " to host " + hostId + " is invalid. Ignoring");
            return false;
        }
        return true;
    }

    // private boolean addNewVm(final String type) {
    //     cloudSimProxy.addNewVm(type);        
    //     return true;
    // }

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

        metricsStorage.updateMetric(
                "vmAllocatedRatioHistory",
                getVmAllocatedRatio());
        metricsStorage.updateMetric(
                "avgCPUUtilizationHistory",
                safeMean(cpuPercentUsage));
        metricsStorage.updateMetric(
                "p90CPUUtilizationHistory", 
                percentileOrZero(cpuPercentUsage, 0.90));
        metricsStorage.updateMetric(
                "avgMemoryUtilizationHistory",
                safeMean(memPercentageUsage));
        metricsStorage.updateMetric(
                "p90MemoryUtilizationHistory", 
                percentileOrZero(memPercentageUsage, 0.90));
        metricsStorage.updateMetric(
                "waitingJobsRatioGlobalHistory",
                waitingJobsRatioGlobal);
        metricsStorage.updateMetric(
                "waitingJobsRatioRecentHistory",
                waitingJobsRatioRecent);
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
        /* reward is the negative cost of running the infrastructure
         * minus any penalties from jobs waiting in the queue
         * minus penalty if action was invalid
        */ 
        final double jobWaitCoef = settings.getRewardJobWaitCoef();
        final double vmCostCoef = settings.getRewardVmCostCoef();
        final double invalidCoef = settings.getRewardInvalidCoef();
        
        final double vmCostPenalty = cloudSimProxy.getRunningCost();
        final double jobWaitPenalty =
        cloudSimProxy.getWaitingJobsCount() 
        * settings.getQueueWaitPenalty() * settings.getSimulationSpeedup();
        final int invalidActionPenalty = (isValid) ? 0 : 1000;
        
        if (!isValid) {
            info("Penalty given to the agent because the selected action was not possible");
        }

        return - jobWaitCoef * jobWaitPenalty 
                - vmCostCoef * vmCostPenalty
                - invalidCoef * invalidActionPenalty;
    }

    public int getStepCount() {
        return stepCount;
    }

    private void resetStepCount() {
        stepCount = 0;
    }

    private void updateMaxCost(double cost) {
        if (cost > maxCost) {
            maxCost = cost;
        }
    }

    private void updateMeanCostPenalty(double costPenalty) {
        meanCostPenalty = (meanCostPenalty * (stepCount - 1) + costPenalty) / stepCount;
    }

    private void updateMeanJobWaitPenalty(double jobWaitPenalty) {
        meanJobWaitPenalty = (meanJobWaitPenalty * (stepCount - 1) + jobWaitPenalty) / stepCount;
    }

    public double getMaxCost() {
        return maxCost;
    }

    private int getValidCount() {
        return validCount;
    }

    private double getMeanJobWaitPenalty() {
        return meanJobWaitPenalty;
    }

    private double getMeanCostPenalty() {
        return meanCostPenalty;
    }

    private void resetMaxCost() {
        maxCost = 0.0;
    }

    private void resetValidCount() {
        validCount = 0;
    }

    private void resetMeanCostPenalty() {
        meanCostPenalty = 0.0;
    }

    private void resetMeanJobWaitPenalty() {
        meanJobWaitPenalty = 0.0;
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

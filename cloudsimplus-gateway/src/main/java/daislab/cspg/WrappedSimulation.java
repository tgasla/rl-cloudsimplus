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
    private static final int HISTORY_LENGTH = 30 * 60; // 30 * 60s = 1800s (30 minutes)
    
    private final List<CloudletDescriptor> initialJobsDescriptors;
    private final List<String> metricsNames = Arrays.asList(
        "vmAllocatedRatio",
        "avgCpuUtilization",
        "p90CpuUtilization",
        // "avgMemoryUtilization",
        // "p90MemoryUtilization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioTimestep"
    );

    private final MetricsStorage metricsStorage = new MetricsStorage(HISTORY_LENGTH, metricsNames);
    private final Gson gson = new Gson();
    private final String identifier;
    private final SimulationSettings settings;
    private final SimulationHistory simulationHistory;
    private CloudSimProxy cloudSimProxy;
    private int stepCount;
    private double jobWaitReward;
    private double utilReward;
    private double invalidReward;
    private int episodeCount = 0;
    private double epJobWaitRewardMean = 0.0;
    private double epUtilRewardMean = 0.0;
    private int epValidCount = 0;
    private long epWaitingJobsCountMax = 0;
    private long epRunningVmsCountMax = 0;
    private CsvWriter unutilizedCsv;
    private CsvWriter unutilizedAllCsv;

    public WrappedSimulation(
        final String identifier,
        final SimulationSettings settings,
        final List<CloudletDescriptor> jobs
    ) {
        this.settings = settings;
        this.identifier = identifier;
        this.initialJobsDescriptors = jobs;
        this.simulationHistory = new SimulationHistory();
        this.unutilizedCsv = null;
        this.unutilizedAllCsv = null;
        
        String[] unutilizedHeader = {"unutilizedOverActiveRatio"};
        unutilizedCsv = new CsvWriter(settings.getJobLogDir(), "unutilized.csv", unutilizedHeader);

        String[] unutilizedAllHeader = {"unutilizedOverAllRatio"};
        unutilizedAllCsv = new CsvWriter(settings.getJobLogDir(), "unutilized_all.csv", unutilizedAllHeader);

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

        if (episodeCount != 1 && unutilizedCsv != null) {
            unutilizedCsv.close();
        }
        if (episodeCount != 1 && unutilizedAllCsv != null) {
            unutilizedAllCsv.close();
        }

        // first attempt to store some memory
        metricsStorage.clear();

        List<Cloudlet> cloudlets = initialJobsDescriptors
            .stream()
            .map(CloudletDescriptor::toCloudlet)
            .collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(
            settings,
            cloudlets,
            episodeCount
        );

        double[] obs = getObservation();

        resetEpJobWaitRewardMean();
        resetEpUtilRewardMean();
        resetEpValidCount();
        resetEpWaitingJobsCountMax();
        resetEpRunningVmsCountMax();

        SimulationStepInfo info = new SimulationStepInfo();

        return new SimulationResetResult(obs, info);
    }

    public void close() {
        info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.getSimulation().terminate();
        }
    }

    public String render() {
        Map<String, double[]> renderedEnv = new HashMap<>();
        for (int i = 0; i < metricsNames.size(); i++) {
            renderedEnv.put(
                metricsNames.get(i),
                metricsStorage.metricValuesAsPrimitives(metricsNames.get(i))
            );
        }
        return gson.toJson(renderedEnv);
    }

    public SimulationStepResult step(final double[] action) {
        // debug("action received");

        if (cloudSimProxy == null) {
            throw new IllegalStateException("Simulation not reset! Please call the reset() function!");
        }

        stepCount++;

        debug("Executing action: " + action[0] + ", " + action[1]);

        long startActionTime = TimeMeasurement.startTiming();
        boolean isValid = executeAction(action);
        if (isValid) {
            epValidCount++;
        }

        long elapsedActionTimeInNs = TimeMeasurement.calculateElapsedTime(startActionTime);
        cloudSimProxy.runFor(settings.getTimestepInterval());

        long startMetricsTime = TimeMeasurement.startTiming();
        collectMetrics();
        long elapsedMetricsTimeInNs = TimeMeasurement.calculateElapsedTime(startMetricsTime);

        boolean done = !cloudSimProxy.isRunning();
        double[] observation = getObservation();
        double reward = calculateReward(isValid);

        simulationHistory.record("action[0]", action[0]);
        simulationHistory.record("action[1]", action[1]);
        simulationHistory.record("reward", reward);
        simulationHistory.record("totalCost", cloudSimProxy.getRunningCost());
        simulationHistory.record(
            "vmExecCount", cloudSimProxy.getBroker().getVmExecList().size());

        if (!cloudSimProxy.isRunning()) {
            simulationHistory.logHistory();
            simulationHistory.reset();
        }

        debug("Step finished (action: " + action[0] + ", " + action[1] + ") is done: " + done
            + " Length of future events queue: " + cloudSimProxy.getNumberOfFutureEvents()
            + " Metrics (s): " + elapsedMetricsTimeInNs / 1_000_000_000d
            + " Action (s): " + elapsedActionTimeInNs / 1_000_000_000d);

        double jobWaitReward = - settings.getRewardJobWaitCoef() * getWaitingJobsRatioGlobal();
        double utilReward = - settings.getRewardUtilizationCoef() * getVmAllocatedRatio();
        double invalidReward = - settings.getRewardInvalidCoef() * (isValid ? 0 : 1);

        updateEpWaitingJobsCountMax(cloudSimProxy.getWaitingJobsCount());
        updateEpRunningVmsCountMax(cloudSimProxy.getBroker().getVmExecList().size());

        updateEpJobWaitRewardMean(-settings.getRewardJobWaitCoef() * jobWaitReward);
        updateEpUtilRewardMean(-settings.getRewardUtilizationCoef() * utilReward);

        debug(" Mean episode job wait reward: " + getEpJobWaitRewardMean()
            + " Mean episode utilization reward: " + getEpUtilRewardMean()
            + " Max episode waiting jobs count: " + getEpWaitingJobsCountMax()
            + " Max episode running vms count: " + getEpRunningVmsCountMax()
            + " Job wait reward: " + jobWaitReward
            + " Utilization reward: " + utilReward
            + " Invalid reward: " + invalidReward
        );
        
        /*
         * METRIC GATHERING CODE START
         */
        List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
        // We could also keep a Map<Host, Integer, Integer>
        //  hostId,  vmsRunning, pesUtilized
        List<long[]> hostMetrics = new ArrayList<>(hostList.size());
        // int[] hostVmsRunningCount = new int[hostList.size()];
        // int[] hostPesUtilized = new int[hostList.size()];
        for (Host host : hostList) {
            hostMetrics.add(
                new long[] {
                    host.getId(),
                    host.getVmList().size(),
                    host.getBusyPesNumber()
                }
            );
        }

        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        // consider adding cores utilized: vm.getPesNumber() - vm.getFreePesNumber()
        //  vmId,    vmPesNumber,  hostId,  jobsRunning 
        List<long[]> vmMetrics = new ArrayList<>(vmList.size());
        for (Vm vm : vmList) {
            vmMetrics.add(
                new long[] {
                    vm.getId(),
                    vm.getPesNumber(),
                    vm.getHost().getId(),
                    vm.getCloudletScheduler().getCloudletList().size()
                }
            );
        }

        // create unutilization log
        if (episodeCount == 1) {
            // TODO: instead of putting if else below, consider .orElse(-1L);
            Long unutilizedCores = vmList
                .parallelStream()
                .map(Vm::getFreePesNumber)
                .reduce(Long::sum)
                .orElse(0L);
            Long runningVmCores = vmList
                .parallelStream()
                .map(Vm::getPesNumber)
                .reduce(Long::sum)
                .orElse(0L);

            // THIS DOES NOT WORK!!!
            // double unutilizedActive;
            // double unutilizedAll;
            // if (vmList.size() == 0) {
            //     unutilizedActive = -1.0;
            //     unutilizedAll = -1.0;
            // }
            // else {
            //     unutilizedActive = (double) (unutilizedCores / runningVmCores);
            //     unutilizedAll = (double) (unutilizedCores / settings.getAvailableCores());
            // }

            if (runningVmCores > 0) {
                Object[] csvRow = {(double) unutilizedCores / runningVmCores};
                unutilizedCsv.writeRow(csvRow);

                csvRow = new Object[] {(double) unutilizedCores / settings.getAvailableCores()};
                unutilizedAllCsv.writeRow(csvRow);
            }
        }

        List<Cloudlet> cloudletList = cloudSimProxy.getBroker()
            .getVmExecList()
            .parallelStream()
            .map(Vm::getCloudletScheduler)
            .map(CloudletScheduler::getCloudletList)
            .flatMap(List::stream)
            .collect(Collectors.toList()
        );

        // consider adding cloudlet.getPesNumber()
        //   jobId,  jobPes,  vmId,    vmType,  hostId
        List<long[]> jobMetrics = new ArrayList<>(cloudletList.size());
        for (Cloudlet cloudlet : cloudletList) {
            jobMetrics.add(
                new long[] {
                    cloudlet.getId(),
                    cloudlet.getPesNumber(),
                    cloudlet.getVm().getId(),
                    cloudlet.getVm().getPesNumber(),
                    cloudlet.getVm().getHost().getId()
                }
            );
        }

        List<Cloudlet> jobsFinishedThisTimestep = cloudSimProxy.getJobsFinishedThisTimestep();
        List<Double> jobWaitTime = new ArrayList<>();
        for (Cloudlet cloudlet : jobsFinishedThisTimestep) {
            jobWaitTime.add(cloudlet.getStartWaitTime());
        }
        /*
         * METRIC GATHERING CODE STOP
         */

        SimulationStepInfo info = new SimulationStepInfo(
            jobWaitReward,
            utilReward,
            invalidReward,
            getEpJobWaitRewardMean(),
            getEpUtilRewardMean(),
            getEpValidCount(),
            hostMetrics,
            vmMetrics,
            jobMetrics,
            new ArrayList<>(0), //jobWaitTime,
            0.0, //unutilizedActive,
            0.0 //unutilizedAll
        );

        return new SimulationStepResult(
            observation,
            reward,
            done,
            info
        );
    }

    private long continuousToDiscrete(
            final double continuous, 
            final long bucketsNum) {
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
            final long discrete =
                (long) Math.min(Math.floor(continuous * bucketsNum), bucketsNum - 1);
        return discrete;
    }

    private boolean executeAction(final double[] action) {

        debug("action is " + action[0] + ", " + action[1]);

        final boolean isValid;
        final long id;
        final int index;
        final int vmTypeIndex;

        // action < 0 destroys the VM with VM.index = index
        if (action[0] < 0) {
            index = (int) continuousToDiscrete(
                Math.abs(action[0]),
                cloudSimProxy.getBroker().getVmExecList().size());

            if (index < 0) {
                debug("No active Vms. Ignoring action...");
                return false;
            }
            debug("translated action[0] = " + index);
            debug("will try to destroy vm with index = " + index);
            isValid = removeVm(index);
            return isValid;
        }

        // action > 0 creates a VM in host host.id = id
        // and Vm.type = action[1]
        else if (action[0] > 0) {
            id = continuousToDiscrete(
                action[0],
                settings.getDatacenterHostsCnt());

            vmTypeIndex = (int) continuousToDiscrete(
                action[1],
                CloudSimProxy.VM_TYPES.length);

            debug("Translated action[0] = " + id);
            debug("Will try to create a new vm at host with id = " 
                    + id + " of type " + CloudSimProxy.VM_TYPES[vmTypeIndex]);
            isValid = addNewVm(CloudSimProxy.VM_TYPES[vmTypeIndex], id);
            return isValid;
        }
        else {
            // action[0] = 0 does nothing
            return true;
        }
    }

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
        double waitingJobsRatioTimestep = getWaitingJobsRatioTimestep();

        metricsStorage.updateMetric(
            "vmAllocatedRatio",
            getVmAllocatedRatio());
        metricsStorage.updateMetric(
            "avgCpuUtilization",
            safeMean(cpuPercentUsage));
        metricsStorage.updateMetric(
            "p90CpuUtilization", 
            percentileOrZero(cpuPercentUsage, 0.90));
        // metricsStorage.updateMetric(
        //     "avgMemoryUtilization",
        //     safeMean(memPercentageUsage));
        // metricsStorage.updateMetric(
            // "p90MemoryUtilization", 
            // percentileOrZero(memPercentageUsage, 0.90));
        metricsStorage.updateMetric(
            "waitingJobsRatioGlobal",
            waitingJobsRatioGlobal);
        metricsStorage.updateMetric(
            "waitingJobsRatioTimestep",
            waitingJobsRatioTimestep);
    }

    private double getWaitingJobsRatioTimestep() {
        final int submittedJobsCountLastInterval =
            cloudSimProxy.getSubmittedJobsCountLastInterval();
        if (submittedJobsCountLastInterval == 0) {
            return 0.0;
        }
        return cloudSimProxy.getWaitingJobsCountLastInterval() 
            / (double) submittedJobsCountLastInterval;
    }

    private double getWaitingJobsRatioGlobal() {
        final int submittedJobsCount = cloudSimProxy.getSubmittedJobsCount();
        if (submittedJobsCount == 0) {
            return 0.0;
        }

        return cloudSimProxy.getWaitingJobsCount() / (double) submittedJobsCount;
    }

    // TODO: rename to datacenter resource usage
    private double getVmAllocatedRatio() {
        return ((double) cloudSimProxy.getNumberOfActiveCores()) / settings.getAvailableCores();
    }

    private double[] getObservation() {
        return new double[] {
            metricsStorage.getLastMetricValue("vmAllocatedRatio"),
            metricsStorage.getLastMetricValue("avgCpuUtilization"),
            metricsStorage.getLastMetricValue("p90CpuUtilization"),
            // metricsStorage.getLastMetricValue("avgMemoryUtilization"),
            // metricsStorage.getLastMetricValue("p90MemoryUtilization"),
            metricsStorage.getLastMetricValue("waitingJobsRatioGlobal"),
            metricsStorage.getLastMetricValue("waitingJobsRatioTimestep")
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
        final double utilizationCoef = settings.getRewardUtilizationCoef();
        final double invalidCoef = settings.getRewardInvalidCoef();
        
        final double jobWaitReward = getWaitingJobsRatioGlobal();
        final double utilReward = getVmAllocatedRatio();
        final int invalidReward = isValid ? 0 : 1;
        
        if (!isValid) {
            info("Penalty given to the agent because the selected action was not possible");
        }

        return - jobWaitCoef * jobWaitReward
                - utilizationCoef * utilReward
                - invalidCoef * invalidReward;
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

    private void updateEpRunningVmsMax(long runningVms) {
        if (runningVms > epRunningVmsCountMax) {
            epRunningVmsCountMax = runningVms;
        }
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
        epUtilRewardMean
                = (epUtilRewardMean * (stepCount - 1) +  utilReward) / stepCount;
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

    public CloudSimProxy getSimulation() {
        return cloudSimProxy;
    }

    public SimulationSettings getSimulationSettings() {
        return settings;
    }
}

package daislab.cspg;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.apache.commons.lang3.text.StrTokenizer;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
// import com.google.gson.Gson;

public class WrappedSimulation {
    private final Logger LOGGER;

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

    // TODO: I should not have it hardcoded here.
    // I should pass a map with <key, value> metrics into the metricsstorage
    // and then calculate the elements of the maps
    // I can simply avoid hardcoding it by first calling collectMetrics and get their lengths. Then
    // calculate maxvmscount and observation array rows and columns
    // In metricstorage initialization I do not need essentialy to give the correct lengths at
    // first.
    private static final int datacenterMetricsCount = 6;
    private static final int hostMetricsCount = 9;
    private static final int vmMetricsCount = 6;
    private static final int jobMetricsCount = 6;

    // private final Gson gson = new Gson();
    private final String identifier;
    private final MetricsStorage metricsStorage;
    private final SimulationHistory simulationHistory;
    private final int hostsCount;
    private final int maxVmsCount;
    private final int observationArrayRows;
    private final int observationArrayColumns;
    private final int minJobPes = 1;
    private CloudSimProxy cloudSimProxy;
    private int currentStep;
    private long epWaitingJobsCountMax = 0;
    private long epRunningVmsCountMax = 0;

    public WrappedSimulation(final String identifier, final List<CloudletDescriptor> jobs) {
        this.identifier = identifier;
        this.initialJobsDescriptors = jobs;

        this.simulationHistory = new SimulationHistory();
        this.hostsCount = Settings.getHostsCount();
        this.maxVmsCount =
                Settings.getHostsCount() * Settings.getHostPes() / Settings.getSmallVmPes();
        final int maxJobsCount = maxVmsCount * Settings.getSmallVmPes() / this.minJobPes;
        this.observationArrayRows = 1 + hostsCount + maxVmsCount + maxJobsCount;
        this.observationArrayColumns = 4;
        // Math.max(hostMetricsCount, Math.max(vmMetricsCount, jobMetricsCount));

        this.metricsStorage = new MetricsStorage(datacenterMetricsCount, hostMetricsCount,
                vmMetricsCount, jobMetricsCount, this.hostsCount, this.maxVmsCount, maxJobsCount);

        this.LOGGER = LoggerFactory.getLogger(getLoggerPrefix());

        LOGGER.info("Creating simulation: {}", identifier);
    }

    public SimulationResetResult reset() {
        LOGGER.info("Reset initiated");
        LOGGER.info("job count: " + initialJobsDescriptors.size());

        resetCurrentStep();

        metricsStorage.clear();

        List<Cloudlet> cloudlets = initialJobsDescriptors.stream()
                .map(CloudletDescriptor::toCloudlet).collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(identifier, cloudlets);

        // gets metric data saved into metricsStorage and concatenates all of them into a 2d array
        double[][] obs = getObservation();
        resetEpisodeStats();
        SimulationStepInfo info = new SimulationStepInfo(identifier);

        return new SimulationResetResult(obs, info);
    }

    public void resetEpisodeStats() {
        resetEpWaitingJobsCountMax();
        resetEpRunningVmsCountMax();
    }

    public void close() {
        LOGGER.info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.terminate();
        }
    }

    public String render() {
        // Map<String, double[]> renderedEnv = new HashMap<>();
        // for (int i = 0; i < metricsNames.size(); i++) {
        // renderedEnv.put(
        // metricsNames.get(i),
        // metricsStorage.metricValuesAsPrimitives(metricsNames.get(i))
        // );
        // }
        // return gson.toJson(renderedEnv);
        return "";
    }

    public void validateSimulationReset() {
        if (cloudSimProxy == null) {
            throw new IllegalStateException(
                    "Simulation not reset! Please call the reset() function before calling step!");
        }
    }

    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();
        this.currentStep++;

        LOGGER.debug("Step {} starts", this.currentStep);

        boolean isValid = executeAction(action);
        cloudSimProxy.runFor(Settings.getTimestepInterval());

        final int[] treeArray = createObservationTreeArray();
        final String dotString = observationTreeArrayToDot(treeArray);
        // gets telemetry data and saves it into metricsStorage
        collectMetrics();

        boolean terminated = !cloudSimProxy.isRunning();
        boolean truncated = !terminated && (currentStep >= Settings.getMaxEpisodeLength());

        // gets metric data saved into metricsStorage and concatenates all of them into a 2d array
        double[][] observation = getObservation();
        double[] rewards = calculateReward(isValid);

        recordSimulationData(action, rewards);

        LOGGER.debug("Step {} finished", this.currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", cloudSimProxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            printAndResetSimulationHistory();
        }

        updateEpisodeStats();
        printEpisodeStatsDebug(rewards);

        SimulationStepInfo info = new SimulationStepInfo(rewards, getCurrentTimestepMetrics(),
                cloudSimProxy.getFinishedJobsWaitTimeLastTimestep(), getUnutilizedVmCoreRatio(),
                dotString);

        return new SimulationStepResult(observation, rewards[0], terminated, truncated, info);
    }

    private List<double[][]> getCurrentTimestepMetrics() {
        List<double[][]> metrics = new ArrayList<>();

        metrics.add(metricsStorage.getHostMetrics());
        metrics.add(metricsStorage.getVmMetrics());
        metrics.add(metricsStorage.getJobMetrics());

        return metrics;
    }

    private List<Cloudlet> getCloudletList() {
        List<Cloudlet> cloudletList = cloudSimProxy.getBroker().getVmExecList().parallelStream()
                .map(Vm::getCloudletScheduler).map(CloudletScheduler::getCloudletList)
                .flatMap(List::stream).collect(Collectors.toList());

        return cloudletList;
    }

    private double getUnutilizedVmCoreRatio() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        Long unutilizedVmCores = getUnutilizedVmCores(vmList);
        Long runningVmCores = getRunningVmCores(vmList);

        return runningVmCores > 0 ? ((double) unutilizedVmCores / runningVmCores) : 0.0;
    }

    private Long getUnutilizedVmCores(List<Vm> vmList) {
        Long unutilizedVmCores =
                vmList.parallelStream().map(Vm::getExpectedFreePesNumber).reduce(0L, Long::sum);

        return unutilizedVmCores;
    }

    private Long getRunningVmCores(List<Vm> vmList) {
        Long runningVmCores = vmList.parallelStream().map(Vm::getPesNumber).reduce(0L, Long::sum);

        return runningVmCores;
    }

    private void printAndResetSimulationHistory() {
        simulationHistory.logHistory();
        simulationHistory.reset();
    }

    private Long getRunningVmsCount() {
        return cloudSimProxy.getBroker().getVmExecList().stream().count();
    }

    private Long getRunningCloudletsCount() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();

        Long runningCloudletCount = vmList.parallelStream().map(Vm::getCloudletScheduler)
                .map(CloudletScheduler::getCloudletExecList).mapToLong(List::size).sum();
        return runningCloudletCount;
    }

    private long vmCountByType(List<Vm> vmList, String type) {
        long filteredVmCount =
                vmList.stream().filter(vm -> type.equals(vm.getDescription())).count();

        return filteredVmCount;
    }

    private double[] collectDatacenterMetrics() {
        double[] datacenterMetrics = new double[] {(double) cloudSimProxy.getAllocatedCores(),
                (double) Settings.getTotalHostCores(), getHostCoresAllocatedToVmsRatio(),
                (double) Settings.getHostsCount(), (double) getRunningVmsCount(),
                (double) getRunningCloudletsCount()};

        return datacenterMetrics;
    }

    private double[][] collectHostMetrics() {
        List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
        // We could also keep a Map<Host, Integer, Integer>
        // hostId, vmsRunning, pesUtilized
        double[][] hostMetrics = new double[hostList.size()][4];

        // int[] hostVmsRunningCount = new int[hostList.size()];
        // int[] hostPesUtilized = new int[hostList.size()];
        for (int i = 0; i < hostList.size(); i++) {
            Host host = hostList.get(i);
            List<Vm> vmList = hostList.get(i).getVmList();
            long smallVmCount = vmCountByType(vmList, "S");
            long mediumVmCount = vmCountByType(vmList, "M");
            long largeVmCount = vmCountByType(vmList, "L");

            hostMetrics[i] = new double[] {host.getId(), host.getVmList().size(), smallVmCount,
                    smallVmCount
                            / (Settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("S")),
                    mediumVmCount,
                    mediumVmCount
                            / (Settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("M")),
                    largeVmCount,
                    largeVmCount
                            / (Settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("L")),
                    host.getBusyPesNumber() / host.getPesNumber()};
        }
        return hostMetrics;
    }

    private double[][] collectVmMetrics() {
        // consider adding cores utilized: vm.getPesNumber() - vm.getFreePesNumber()
        // vmId, vmPesNumber, hostId, jobsRunning
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        double[][] vmMetrics = new double[vmList.size()][1];
        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            vmMetrics[i] = new double[] {vm.getId(), vm.getHost().getId(),
                    vm.getCloudletScheduler().getCloudletList().size(),
                    vm.getCloudletScheduler().getCloudletExecList().size(),
                    vm.getCloudletScheduler().getCloudletWaitingList().size(),
                    (vm.getPesNumber() - vm.getFreePesNumber()) / vm.getPesNumber()};
        }
        return vmMetrics;
    }

    private double[][] collectJobMetrics() {
        List<Cloudlet> cloudletList = getCloudletList();
        // jobId, jobPes, vmId, vmType, hostId
        double[][] jobMetrics = new double[cloudletList.size()][1];
        for (int i = 0; i < cloudletList.size(); i++) {
            Cloudlet cloudlet = cloudletList.get(i);
            jobMetrics[i] = new double[] {cloudlet.getId(), cloudlet.getPesNumber(),
                    cloudlet.getVm().getId(), cloudlet.getVm().getPesNumber(),
                    cloudlet.getVm().getHost().getId(),
                    cloudlet.getPesNumber() / cloudlet.getVm().getPesNumber()};
        }
        return jobMetrics;
    }

    // private double[] listOfArraysToArray(List<double[]> listOfArrays) {
    // final int listSize = listOfArrays.size();
    // final int arraySize = listOfArrays.get(0).length;
    // // we know that each array in the list ahs the same size
    // // so we calculate the final array size just by the size of the list
    // // and the first array length
    // double[] resultArray = new double[listSize * arraySize];

    // int currentIndex = 0;
    // for (double[] array : listOfArrays) {
    // System.arraycopy(array, 0, resultArray, currentIndex, array.length);
    // currentIndex += array.length;
    // }

    // return resultArray;
    // }

    private void printEpisodeStatsDebug(double[] reward) {
        LOGGER.debug("\n==================== Episode stats so far ===================="
                + "\nEpisode Statistics:\nMax waiting jobs count: " + getEpWaitingJobsCountMax()
                + "\nMax running vms count in the episode: " + getEpRunningVmsCountMax()
                + "\n===================================================="
                + "\nTimestep statistics:\nJob wait reward: " + reward[1]
                + "\nRunning VM cores reward: " + reward[2] + "\nUnutilized VM cores reward: "
                + reward[3] + "\nInvalid reward: " + reward[4]
                + "\n==============================================================");
    }

    private void updateEpisodeStats() {
        updateEpWaitingJobsCountMax(cloudSimProxy.getWaitingJobsCount());
        updateEpRunningVmsCountMax(cloudSimProxy.getBroker().getVmExecList().size());
    }

    // private long continuousToDiscrete(final double continuous, final long bucketsNum) {
    // /*
    // * Explanation:
    // * floor(continuous * bucketsNum) will give you the discrete value
    // * but, in case of cont = 1, then the discrete value
    // * will be equal to bucketsNum.
    // * However, we want to map the continuous value to the
    // * range of [0,bucketsNum-1].
    // * So, Math.min ensures that the maximum allowed
    // * discrete value will be bucketsNum-1.
    // */
    // final long discrete = (long) Math.min(Math.floor(continuous * bucketsNum), bucketsNum - 1);
    // return discrete;
    // }

    private void recordSimulationData(int[] action, double[] reward) {
        simulationHistory.record("action[0]", action[0]);
        simulationHistory.record("action[1]", action[1]);
        simulationHistory.record("action[2]", action[2]);
        simulationHistory.record("action[3]", action[3]);
        simulationHistory.record("totalReward", reward[0]);
        simulationHistory.record("jobWaitReward", reward[1]);
        simulationHistory.record("runningVmCoresReward", reward[2]);
        simulationHistory.record("unutilizedVmCoresReward", reward[3]);
        simulationHistory.record("invalidReward", reward[4]);
        simulationHistory.record("vmExecCount", cloudSimProxy.getBroker().getVmExecList().size());
        // simulationHistory.record("totalCost", cloudSimProxy.getRunningCost());
    }

    private boolean executeAction(final int[] action) {

        final boolean isValid;

        LOGGER.debug("The action is [{}, {}, {}, {}]", action[0], action[1], action[2], action[3]);

        // [action, id, type]
        // action = {0: do nothing, 1: create vm, 2: destroy vm}
        // id = {hostId to place new vm (when action = 1), vmId to terminate (when action = 2)
        // type = {0: small, 1: medium, 2: large} (relevant only when action = 1)

        if (action[0] == 1) {
            final int hostId = action[1];
            final int vmTypeIndex = action[3];
            isValid = addNewVm(Settings.VM_TYPES[vmTypeIndex], hostId);
            return isValid;
        }

        else if (action[0] == 2) {
            final int vmIndex = action[2];
            isValid = removeVm(vmIndex);
            return isValid;
        }

        return true;
    }

    // final long id;
    // final int index;
    // final int vmTypeIndex;

    // // action < 0 destroys the VM with VM.index = index
    // if (action[0] < 0) {
    // index = (int) continuousToDiscrete(
    // Math.abs(action[0]),
    // cloudSimProxy.getBroker().getVmExecList().size());

    // if (index < 0) {
    // debug("No active Vms. Ignoring action...");
    // return false;
    // }
    // debug("translated action[0] = " + index);
    // debug("will try to destroy vm with index = " + index);
    // isValid = removeVm(index);
    // return isValid;
    // }

    // // action > 0 creates a VM in host host.id = id
    // // and Vm.type = action[1]
    // else if (action[0] > 0) {
    // id = continuousToDiscrete(
    // action[0],
    // Settings.getDatacenterHostsCnt());

    // vmTypeIndex = (int) continuousToDiscrete(
    // action[1],
    // Settings.VM_TYPES.length);

    // debug("Translated action[0] = " + id);
    // debug("Will try to create a new vm at host with id = "
    // + id + " of type " + Settings.VM_TYPES[vmTypeIndex]);
    // isValid = addNewVm(Settings.VM_TYPES[vmTypeIndex], id);
    // return isValid;
    // }
    // else {
    // // action[0] = 0 does nothing
    // return true;
    // }

    private boolean removeVm(final int index) {
        if (!cloudSimProxy.removeVm(index)) {
            LOGGER.debug("Removing a VM with index {} action is invalid. Ignoring.", index);
            return false;
        }
        return true;
    }

    // adds a new vm to the host with hostid if possible
    private boolean addNewVm(final String type, final long hostId) {
        if (!cloudSimProxy.addNewVm(type, hostId)) {
            LOGGER.debug("Adding a VM of type {} to host {} is invalid. Ignoring", type, hostId);
            return false;
        }
        return true;
    }

    private void collectMetrics() {
        metricsStorage.setDatacenterMetrics(collectDatacenterMetrics());
        metricsStorage.setHostMetrics(collectHostMetrics());
        metricsStorage.setVmMetrics(collectVmMetrics());
        metricsStorage.setJobMetrics(collectJobMetrics());
    }

    // private void collectMetrics() {
    // double[] cpuPercentUsage = cloudSimProxy.getVmCpuUsage();
    // Arrays.sort(cpuPercentUsage);

    // double[] memPercentageUsage = cloudSimProxy.getVmMemoryUsage();
    // Arrays.sort(memPercentageUsage);

    // double waitingJobsRatioGlobal = getWaitingJobsRatio();
    // double waitingJobsRatioLastTimestep = getWaitingJobsRatioLastTimestep();

    // metricsStorage.updateMetric(
    // "hostCoresAllocatedToVmsRatio",
    // getHostCoresAllocatedToVmsRatio());
    // metricsStorage.updateMetric(
    // "avgCpuUtilization",
    // safeMean(cpuPercentUsage));
    // metricsStorage.updateMetric(
    // "p90CpuUtilization",
    // percentileOrZero(cpuPercentUsage, 0.90));
    // metricsStorage.updateMetric(
    // "avgMemoryUtilization",
    // safeMean(memPercentageUsage));
    // metricsStorage.updateMetric(
    // "p90MemoryUtilization",
    // percentileOrZero(memPercentageUsage, 0.90));
    // metricsStorage.updateMetric(
    // "waitingJobsRatioGlobal",
    // waitingJobsRatioGlobal);
    // metricsStorage.updateMetric(
    // "waitingJobsRatioTimestep",
    // waitingJobsRatioTimestep);
    // }

    private double getWaitingJobsRatio() {
        final long arrivedJobsCount = cloudSimProxy.getArrivedJobsCount();

        return arrivedJobsCount > 0
                ? cloudSimProxy.getWaitingJobsCount() / (double) arrivedJobsCount
                : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        return ((double) cloudSimProxy.getAllocatedCores()) / Settings.getTotalHostCores();
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
    private double[][] getColumns(final double[][] matrix, final int[] columnIndices) {
        int numRows = matrix.length;
        int numCols = columnIndices.length;
        double[][] result = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                result[i][j] = matrix[i][columnIndices[j]];
            }
        }
        return result;
    }

    private String observationTreeArrayToDot(int[] state) {
        StringBuilder dotString =
                new StringBuilder("graph {\nd0 [label=").append(state[0]).append("] ");
        StringBuilder dc = new StringBuilder("d0");
        int hostNumIdx = 1;
        int hostCoreIdx = hostNumIdx + 1;
        // loop over the hosts
        for (int i = 0; i < state[hostNumIdx]; i++) {
            StringBuilder host = new StringBuilder("h").append(i);
            dotString.append("\n").append(host).append(" [label=").append(state[hostCoreIdx])
                    .append("] ");
            dotString.append(dc).append("--").append(host);
            int vmNumIdx = hostCoreIdx + 1;
            int vmCoreIdx = vmNumIdx + 1;
            // loop over the vms
            for (int j = 0; j < state[vmNumIdx]; j++) {
                StringBuilder vm = new StringBuilder(host).append("v").append(j);
                dotString.append("\n").append(vm).append(" [label=").append(state[vmCoreIdx])
                        .append("] ");
                dotString.append(host).append("--").append(vm);
                int jobNumIdx = vmCoreIdx + 1;
                int jobCoreIdx = jobNumIdx + 1;
                // loop over the jobs
                for (int k = 0; k < state[jobNumIdx]; k++) {
                    StringBuilder job = new StringBuilder(vm).append("j").append(k);
                    dotString.append("\n").append(job).append(" [label=").append(state[jobCoreIdx])
                            .append("] ");
                    dotString.append(vm).append("--").append(job);
                    jobCoreIdx += 2;
                }
                vmCoreIdx = jobCoreIdx;
            }
            hostCoreIdx = vmCoreIdx;
        }
        dotString.append("\n}");
        return dotString.toString();
    }

    private int[] createObservationTreeArray() {
        final int hostsNum = Settings.getHostsCount();
        final int vmsNum = getRunningVmsCount().intValue();
        final int jobsNum = getRunningCloudletsCount().intValue();
        final int[] treeArray = new int[2 + 2 * hostsNum + 2 * vmsNum + 2 * jobsNum];

        final int totalDatacenterCores = (int) Settings.getDatacenterCores();
        final List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
        treeArray[0] = totalDatacenterCores;
        treeArray[1] = hostsNum;
        int currentIndex = 2;
        for (int i = 0; i < hostsNum; i++) {
            final Host host = hostList.get(i);
            final List<Vm> vmList = host.getVmList();
            treeArray[currentIndex++] = (int) host.getPesNumber();
            treeArray[currentIndex++] = vmList.size();
            for (int j = 0; j < vmList.size(); j++) {
                final Vm vm = vmList.get(j);
                final List<Cloudlet> jobList = vm.getCloudletScheduler().getCloudletList();
                treeArray[currentIndex++] = (int) vm.getPesNumber();
                treeArray[currentIndex++] = jobList.size();
                for (int k = 0; k < jobList.size(); k++) {
                    final Cloudlet cloudlet = jobList.get(k);
                    treeArray[currentIndex++] = (int) cloudlet.getPesNumber();
                    treeArray[currentIndex++] = 0; // jobs do not have children
                }
            }
        }
        // System.out.println("obstreearray");
        // for (int i = 0; i < treeArray.length; i++) {
        // System.out.println(treeArray[i]);
        // }
        return treeArray;
    }

    private double[][] getObservation() {
        // here we get some vertical subarrays of the metrics. The whole array of metrics are only
        // used to pass the info to python and print to csv files then.
        // Vertical because rows in the arrays represent the hosts, vms, jobs etc. So, we take a
        // specific subset of features (columns) for every host, vm or job.

        final double[] datacenterMetrics = new double[] {metricsStorage.getDatacenterMetrics()[2]};
        final double[][] hostMetrics =
                getColumns(metricsStorage.getHostMetrics(), new int[] {3, 5, 7, 8});
        final double[][] vmMetrics = getColumns(metricsStorage.getVmMetrics(), new int[] {5});
        final double[][] jobMetrics = getColumns(metricsStorage.getJobMetrics(), new int[] {5});
        final double[][] observation = new double[observationArrayRows][observationArrayColumns];

        int currentRow = 0;

        for (int j = 0; j < datacenterMetrics.length; j++) {
            observation[currentRow][j] = datacenterMetrics[j];
        }
        currentRow++;

        for (int i = 0; i < hostMetrics.length; i++) {
            for (int j = 0; j < hostMetrics[i].length; j++) {
                observation[currentRow][j] = hostMetrics[i][j];
            }
            currentRow++;
        }

        for (int i = 0; i < vmMetrics.length; i++) {
            for (int j = 0; j < vmMetrics[i].length; j++) {
                observation[currentRow][j] = vmMetrics[i][j];
            }
            currentRow++;
        }

        currentRow = 1 + this.hostsCount + this.maxVmsCount;

        for (int i = 0; i < jobMetrics.length; i++) {
            for (int j = 0; j < jobMetrics[i].length; j++) {
                observation[currentRow][j] = jobMetrics[i][j];
            }
            currentRow++;
        }

        return observation;
    }

    private double[] calculateReward(final boolean isValid) {
        double[] rewards = new double[5];
        /*
         * reward is the negative cost of running the infrastructure minus any penalties from jobs
         * waiting in the queue minus penalty if action was invalid
         */

        final double jobWaitCoef = Settings.getRewardJobWaitCoef();
        final double runningVmCoresCoef = Settings.getRewardRunningVmCoresCoef();
        final double unutilizedVmCoresCoef = Settings.getRewardUnutilizedVmCoresCoef();
        final double invalidCoef = Settings.getRewardInvalidCoef();

        final double jobWaitReward = -jobWaitCoef * getWaitingJobsRatio();
        final double runningVmCoresReward = -runningVmCoresCoef * getHostCoresAllocatedToVmsRatio();
        final double unutilizedVmCoresReward = -unutilizedVmCoresCoef * getUnutilizedVmCoreRatio();
        final double invalidReward = -invalidCoef * (isValid ? 0 : 1);

        final double totalReward =
                jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward + invalidReward;

        LOGGER.debug("totalReward" + totalReward);
        LOGGER.debug("jobWaitReward:" + jobWaitReward);
        LOGGER.debug("runningVmCoresReward:" + runningVmCoresReward);
        LOGGER.debug("unutilizedVmCoresReward" + unutilizedVmCoresReward);
        LOGGER.debug("invalidReward:" + invalidReward);

        rewards[0] = totalReward;
        rewards[1] = jobWaitReward;
        rewards[2] = runningVmCoresReward;
        rewards[3] = unutilizedVmCoresReward;
        rewards[4] = invalidReward;

        if (!isValid) {
            LOGGER.debug("Penalty given to the agent because the selected action was not possible");
        }
        return rewards;
    }

    public int getCurrentStep() {
        return currentStep;
    }

    private void resetCurrentStep() {
        currentStep = 0;
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

    public String getIdentifier() {
        return identifier;
    }

    public void seed() {
        // there is no randomness so far...
    }

    public double clock() {
        return cloudSimProxy.clock();
    }

    public void printJobStats() {
        cloudSimProxy.printJobStats();
    }

    private String getLoggerPrefix() {
        return WrappedSimulation.class.getSimpleName() + ": " + identifier;
    }
}

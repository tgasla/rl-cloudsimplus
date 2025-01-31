package daislab.cspg;

import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
// import com.google.gson.Gson;

public class WrappedSimulation {
    private final Logger LOGGER = LoggerFactory.getLogger(WrappedSimulation.class.getSimpleName());

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

    // I should pass a map with <key, value> metrics into the metricsstorage
    // and then calculate the elements of the maps
    // I can simply avoid hardcoding it by first calling collectMetrics and get
    // their lengths. Then
    // calculate maxvmscount and observation array rows and columns
    // In metricstorage initialization I do not need essentialy to give the correct
    // lengths at
    // first.
    // TODO: I should not have it hardcoded here.
    // private static final int datacenterMetricsCount = 6;
    // private static final int hostMetricsCount = 9;
    // private static final int vmMetricsCount = 6;
    // private static final int jobMetricsCount = 6;

    // private final Gson gson = new Gson();
    private final String identifier;
    private final SimulationSettings settings;
    private CloudSimProxy cloudSimProxy;
    private int currentStep;
    private int bestEpisodeReward;
    private int currentEpisodeReward;
    // private final MetricsStorage metricsStorage;
    // private final SimulationHistory simulationHistory;
    // private final int maxVms;
    // private final int maxHosts;
    // private final int observationArrayRows;
    // private final int observationArrayColumns;
    // private final int minJobPes = 1;
    // private long epWaitingJobsCountMax = 0;
    // private long epRunningVmsCountMax = 0;

    public WrappedSimulation(final String identifier, final SimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        this.identifier = identifier;
        this.settings = settings;
        initialJobsDescriptors = jobs;
        bestEpisodeReward = -Integer.MAX_VALUE;

        // simulationHistory = new SimulationHistory();
        // maxHosts = settings.getMaxHosts();
        // maxVms = maxHosts * settings.getHostPes() / settings.getSmallVmPes();
        // final int maxJobsCount = maxVms * settings.getSmallVmPes() / minJobPes;
        // observationArrayRows = 1 + maxHosts + maxVms + maxJobsCount;
        // observationArrayColumns = 4;
        // Math.max(hostMetricsCount, Math.max(vmMetricsCount, jobMetricsCount));

        // metricsStorage = new MetricsStorage(datacenterMetricsCount, hostMetricsCount,
        // vmMetricsCount, jobMetricsCount, maxHosts, maxVms, maxJobsCount);

        LOGGER.info("Creating simulation: {}", identifier);
    }

    // public void resetEpisodeStats() {
    // resetEpWaitingJobsCountMax();
    // resetEpRunningVmsCountMax();
    // }

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
        // return gson.toJson(renderedEnv)
        return "";
    }

    public void validateSimulationReset() {
        if (cloudSimProxy == null) {
            throw new IllegalStateException(
                    "Simulation not reset! Please call the reset() function before calling step!");
        }
    }

    public SimulationResetResult reset(final long seed) {
        // ignoring seed for now
        LOGGER.info("Reset initiated");
        LOGGER.info("job count: " + initialJobsDescriptors.size());
        // LOGGER.info(settings.getCloudletToDcAssignmentPolicy());

        this.currentStep = 0;
        this.currentEpisodeReward = 0;
        // resetEpisodeStats(); // TEMPORARILY DISABLED FOR OPTIMIZATION
        // metricsStorage.clear();
        // simulationHistory.reset();

        List<Cloudlet> cloudlets = initialJobsDescriptors.stream()
                .map(CloudletDescriptor::toCloudlet).collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(settings, cloudlets);

        SimulationStepInfo info = new SimulationStepInfo(0, 0, 0, new ArrayList<>());

        Observation observation =
                new Observation(getInfrastructureObservation(), getJobsWaitingObservation());

        return new SimulationResetResult(observation, info);
    }

    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();
        currentStep++;
        LOGGER.info("Step {} starting", currentStep);

        final double[] ratios = executeCustomCloudletToDcAction(action);
        // switch (settings.getVmAllocationPolicy()) {
        // case "rl", "fromfile" -> {
        // ratios = executeCustomVmManagementAction(action);
        // }
        // case "rule-based" -> cloudSimProxy.executeRuleBasedAction();
        // default -> throw new IllegalArgumentException(
        // "Unexpected value: " + settings.getVmAllocationPolicy());
        // };

        // final boolean isValid = actionResult[0] != -1;
        // final boolean isValid = true;

        cloudSimProxy.runOneTimestep();

        // Temporarily disabled for optimization. Do not create the dot string from the
        // tree array
        // for every timestep!
        // final TreeArray treeArray = new TreeArray(getObservationAsTreeArray());
        // final String dotString = treeArray.toDot();

        // gets telemetry data and saves it into metricsStorage
        // TEMPORARILY DISABLED FOR OPTIMIZATION
        // updateMetrics();
        /////////////////////////////////////////////////

        boolean terminated = !cloudSimProxy.isRunning();
        boolean truncated = !terminated && (currentStep >= settings.getMaxEpisodeLength());

        double reward = calculateReward(ratios[0], ratios[1], ratios[2]);

        this.currentEpisodeReward += reward;

        // TEMPORARILY DISABLED FOR OPTIMIZATION
        // recordSimulationData(action, rewards);

        LOGGER.info("Step {} finished", currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", cloudSimProxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    cloudSimProxy.getBroker().getCloudletFinishedList().size(),
                    initialJobsDescriptors.size());
            if (currentEpisodeReward > bestEpisodeReward) {
                bestEpisodeReward = currentEpisodeReward;
                LOGGER.info("New best episode reward: {}", bestEpisodeReward);
            }
            // for (double jobWaitTime : cloudSimProxy.getJobsFinishedWaitTimes()) {
            // LOGGER.info("{}", jobWaitTime);
            // }
            // simulationHistory.logHistory();
        }

        // DISABLED FOR OPTIMIZATION
        // updateEpisodeStats();
        // printEpisodeStatsDebug(rewards);

        // OLD INFO, SIMPLIFIED FOR OPTIMIZATION
        // SimulationStepInfo info = new SimulationStepInfo(rewards,
        // getCurrentTimestepMetrics(),
        // cloudSimProxy.getFinishedJobsWaitTimeLastTimestep(),
        // getUnutilizedVmCoreRatio(),
        // getInfrastructureObservation());

        final List<Double> jobWaitTime = cloudSimProxy.getFinishedJobsWaitTimeLastTimestep();

        SimulationStepInfo info =
                new SimulationStepInfo(ratios[0], ratios[1], ratios[2], jobWaitTime);

        // Observation observation =
        // new Observation(getInfrastructureObservation(),
        // getJobCoresWaitingObservation());
        Observation observation =
                new Observation(getInfrastructureObservation(), getJobsWaitingObservation());

        return new SimulationStepResult(observation, reward, terminated, truncated, info);
    }

    // private List<double[][]> getCurrentTimestepMetrics() {
    // List<double[][]> metrics = new ArrayList<>();

    // metrics.add(metricsStorage.getHostMetrics());
    // metrics.add(metricsStorage.getVmMetrics());
    // metrics.add(metricsStorage.getJobMetrics());

    // return metrics;
    // }

    // private List<Cloudlet> getCloudletList() {
    // List<Cloudlet> cloudletList =
    // cloudSimProxy.getBroker().getVmExecList().parallelStream()
    // .map(Vm::getCloudletScheduler).map(CloudletScheduler::getCloudletList)
    // .flatMap(List::stream).collect(Collectors.toList());

    // return cloudletList;
    // }

    double getUnutilizedVmCoreRatio() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        Long unutilizedVmCores = getUnutilizedVmCores(vmList);
        Long runningVmCores = getRunningVmCores(vmList);

        return runningVmCores > 0 ? ((double) unutilizedVmCores / runningVmCores) : 0.0;
    }

    Long getUnutilizedVmCores(List<Vm> vmList) {
        Long unutilizedVmCores =
                vmList.parallelStream().map(Vm::getExpectedFreePesNumber).reduce(0L, Long::sum);

        return unutilizedVmCores;
    }

    private Long getRunningVmCores(List<Vm> vmList) {
        Long runningVmCores = vmList.parallelStream().map(Vm::getPesNumber).reduce(0L, Long::sum);

        return runningVmCores;
    }

    // private Long getRunningVmsCount() {
    // return cloudSimProxy.getBroker().getVmExecList().stream().count();
    // }

    // private Long getRunningCloudletsCount() {
    // List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();

    // Long runningCloudletCount =
    // vmList.parallelStream().map(Vm::getCloudletScheduler)
    // .map(CloudletScheduler::getCloudletExecList).mapToLong(List::size).sum();
    // return runningCloudletCount;
    // }

    // private long vmCountByType(List<Vm> vmList, String type) {
    // long filteredVmCount =
    // vmList.stream().filter(vm -> type.equals(vm.getDescription())).count();

    // return filteredVmCount;
    // }

    // private double[] collectDatacenterMetrics() {
    // double[] datacenterMetrics = new double[] {(double)
    // cloudSimProxy.getAllocatedCores(),
    // (double) settings.getTotalHostCores(), getHostCoresAllocatedToVmsRatio(),
    // (double) settings.getHostsCount(), (double) getRunningVmsCount(),
    // (double) getRunningCloudletsCount()};

    // return datacenterMetrics;
    // }

    // private double[][] collectHostMetrics() {
    // List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
    // // We could also keep a Map<Host, Integer, Integer>
    // // hostId, vmsRunning, pesUtilized
    // double[][] hostMetrics = new double[hostList.size()][4];

    // // int[] hostVmsRunningCount = new int[hostList.size()];
    // // int[] hostPesUtilized = new int[hostList.size()];
    // for (int i = 0; i < hostList.size(); i++) {
    // Host host = hostList.get(i);
    // List<Vm> vmList = hostList.get(i).getVmList();
    // long smallVmCount = vmCountByType(vmList, "S");
    // long mediumVmCount = vmCountByType(vmList, "M");
    // long largeVmCount = vmCountByType(vmList, "L");

    // hostMetrics[i] = new double[] {host.getId(), host.getVmList().size(),
    // smallVmCount,
    // smallVmCount
    // / (settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("S")),
    // mediumVmCount,
    // mediumVmCount
    // / (settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("M")),
    // largeVmCount,
    // largeVmCount
    // / (settings.getHostPes() / cloudSimProxy.getVmCoreCountByType("L")),
    // host.getBusyPesNumber() / host.getPesNumber()};
    // }
    // return hostMetrics;
    // }

    // private double[][] collectVmMetrics() {
    // // consider adding cores utilized: vm.getPesNumber() - vm.getFreePesNumber()
    // // vmId, vmPesNumber, hostId, jobsRunning
    // List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
    // double[][] vmMetrics = new double[vmList.size()][1];
    // for (int i = 0; i < vmList.size(); i++) {
    // Vm vm = vmList.get(i);
    // vmMetrics[i] = new double[] {vm.getId(), vm.getHost().getId(),
    // vm.getCloudletScheduler().getCloudletList().size(),
    // vm.getCloudletScheduler().getCloudletExecList().size(),
    // vm.getCloudletScheduler().getCloudletWaitingList().size(),
    // (vm.getPesNumber() - vm.getFreePesNumber()) / vm.getPesNumber()};
    // }
    // return vmMetrics;
    // }

    // private double[][] collectJobMetrics() {
    // List<Cloudlet> cloudletList = getCloudletList();
    // // jobId, jobPes, vmId, vmType, hostId
    // double[][] jobMetrics = new double[cloudletList.size()][1];
    // for (int i = 0; i < cloudletList.size(); i++) {
    // Cloudlet cloudlet = cloudletList.get(i);
    // jobMetrics[i] = new double[] {cloudlet.getId(), cloudlet.getPesNumber(),
    // cloudlet.getVm().getId(), cloudlet.getVm().getPesNumber(),
    // cloudlet.getVm().getHost().getId(),
    // cloudlet.getPesNumber() / cloudlet.getVm().getPesNumber()};
    // }
    // return jobMetrics;
    // }

    // private void printEpisodeStatsDebug(double[] reward) {
    // LOGGER.debug("Printing Episode stats:"
    // + "\n==================== Episode stats so far ===================="
    // + "\nEpisode Statistics:\nMax waiting jobs count: " +
    // getEpWaitingJobsCountMax()
    // + "\nMax running vms count in the episode: " + getEpRunningVmsCountMax()
    // + "\n=============================================================="
    // + "\nTimestep statistics:\nJob wait reward: " + reward[1]
    // + "\nRunning VM cores reward: " + reward[2] + "\nUnutilized VM cores reward:
    // "
    // + reward[3] +
    // "\n==============================================================");
    // }

    // private void updateEpisodeStats() {
    // updateEpWaitingJobsCountMax(cloudSimProxy.getNotYetRunningJobsCount());
    // updateEpRunningVmsCountMax(cloudSimProxy.getBroker().getVmExecList().size());
    // }

    // private long continuousToDiscrete(final double continuous, final long
    // bucketsNum) {
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
    // final long discrete = (long) Math.min(Math.floor(continuous * bucketsNum),
    // bucketsNum - 1);
    // return discrete;
    // }

    // private void recordSimulationData(int[] action, double[] reward) {
    // simulationHistory.record("action[0]", action[0]);
    // simulationHistory.record("action[1]", action[1]);
    // simulationHistory.record("action[2]", action[2]);
    // simulationHistory.record("action[3]", action[3]);
    // simulationHistory.record("totalReward", reward[0]);
    // simulationHistory.record("jobWaitReward", reward[1]);
    // simulationHistory.record("runningVmCoresReward", reward[2]);
    // simulationHistory.record("unutilizedVmCoresReward", reward[3]);
    // simulationHistory.record("invalidReward", reward[4]);
    // simulationHistory.record("vmExecCount",
    // cloudSimProxy.getBroker().getVmExecList().size());
    // // simulationHistory.record("totalCost", cloudSimProxy.getRunningCost()); #
    // TODO: total cost is not properly calculated, fix it
    // }

    // private Vm getFirstAvailableVmOfDcForCloudlet(final int targetDcId, final
    // Cloudlet cloudlet)
    // {
    // List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
    // List<Cloudlet> cloudletList =
    // cloudSimProxy.getBroker().getCloudletCreatedList();
    // Map<Vm, Integer> vmUsedCoresMap =
    // vmList.stream().collect(Collectors.toMap(vm -> vm, vm -> (int)
    // cloudletList.stream()
    // .filter(c -> c.getVm() == vm).mapToLong(Cloudlet::getPesNumber).sum()));

    // for (Vm vm : vmList) {
    // // Filter cloudlets with the target VM
    // int expectedFreeVmPes = (int) vm.getPesNumber() - vmUsedCoresMap.get(vm);
    // final int dcId = (int) vm.getHost().getDatacenter().getId();
    // if (dcId == targetDcId && vm.isSuitableForCloudlet(cloudlet)
    // && cloudlet.getPesNumber() <= expectedFreeVmPes) {
    // return vm;
    // }
    // }
    // return Vm.NULL;
    // }

    // This function does not take into account the cores that will not be available
    // because of
    // the cloudlets that have been assigned to a vm but not yet submitted.
    // private Vm getFirstAvailableVmOfDcForCloudlet(final int targetDcId, final
    // Cloudlet cloudlet)
    // {
    // List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
    // // LOGGER.info("VMs running when trying to find binding: {}", vmList.size());
    // for (Vm vm : vmList) {
    // final int dcId = (int) vm.getHost().getDatacenter().getId();
    // if (dcId == targetDcId && vm.isSuitableForCloudlet(cloudlet)
    // && vm.getExpectedFreePesNumber() >= cloudlet.getPesNumber()) {
    // LOGGER.info("Found VM in DC {}", dcId);
    // return vm;
    // }
    // }
    // return Vm.NULL;
    // }

    private Vm getMostFreeVmOfDcForCloudlet(final int targetDcId, final Cloudlet cloudlet) {
        long maxExpectedFreePes = 0;
        Vm mostFreeVm = Vm.NULL;
        final double targetTime = cloudSimProxy.calculateTargetTime();
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        List<Cloudlet> cloudletList = cloudSimProxy.getJobsToSubmitAtThisTimestep(targetTime);
        // Map<Vm, Long> vmUsedCoresMap = new HashMap<>();
        // for (Vm vm : vmList) {
        // vmUsedCoresMap.put(vm, 0L);
        // for (Cloudlet c : cloudletList) {
        // if (c.getVm() != Vm.NULL && c.getVm() != null && c.getVm() == vm) {
        // vmUsedCoresMap.put(vm, vmUsedCoresMap.get(vm) + c.getPesNumber());
        // }
        // }
        // }
        Map<Vm, Long> expectedToUseVmPesMap =
                vmList.stream().collect(Collectors.toMap(vm -> vm, vm -> cloudletList.stream()
                        .filter(c -> c.getVm() == vm).mapToLong(Cloudlet::getPesNumber).sum()));
        // Map<Long, Long> vmFreeCoresMap = vmList.stream()
        // .collect(Collectors.toMap(vm -> vm.getId(), vm ->
        // vm.getExpectedFreePesNumber()));
        // LOGGER.debug("{}: {}", clock(), expectedToUseVmPesMap.toString());
        // LOGGER.info("VMs running when trying to find binding: {}", vmList.size());
        for (Vm vm : vmList) {
            final int dcId = (int) vm.getHost().getDatacenter().getId();
            final long usedVmPes = vm.getCloudletScheduler().getCloudletList().stream()
                    .mapToLong(Cloudlet::getPesNumber).sum();
            // this may get negative but it is ok because it will also count the cloudlets
            // that are in some vms queue, so we get an estimation of how much overloaded it
            // is.
            // We get the vm that will have maximum expected free cores
            final long expectedFreePes =
                    vm.getPesNumber() - usedVmPes - expectedToUseVmPesMap.get(vm);
            // final long expectedFreePes =
            // Math.max(0, vm.getExpectedFreePesNumber() - vmUsedCoresMap.get(vm));
            // LOGGER.debug("{}: VM {} has {} expected free cores", clock(), vm.getId(),
            // expectedFreePes);
            if (dcId == targetDcId && vm.isSuitableForCloudlet(cloudlet)
                    && expectedFreePes >= cloudlet.getPesNumber()) {
                if (expectedFreePes > maxExpectedFreePes) {
                    maxExpectedFreePes = expectedFreePes;
                    mostFreeVm = vm;
                }
            }
        }

        LOGGER.debug("{}: Selecting VM {} for cloudlet {} with {} expected free cores",

                clock(), mostFreeVm.getId(), cloudlet.getId(), maxExpectedFreePes);
        return mostFreeVm;
    }

    private double calculateQualityOfPlacement(final int dcId, final Cloudlet job) {
        final String datacenterType =
                ((DatacenterWithType) cloudSimProxy.getDatacenterById(dcId)).getType();
        // jobSensitivity - 0: tolerant, 1: moderate, 2: critical
        final int jobSensitivity = ((CloudletWithLocation) job).getDelaySensitivity();
        if (jobSensitivity == 0 | datacenterType.equals("micro")
                | (datacenterType.equals("edge") && jobSensitivity == 1)) {
            return 1.0;
        }
        if (datacenterType.equals("edge") && jobSensitivity == 2) {
            return 0.5;
        }
        return 0.0;
    }

    private double[] executeCustomCloudletToDcAction(final int[] action) {
        return switch (settings.getCloudletToDcAssignmentPolicy()) {
            case "rl" -> executeRlCloudletToDcAction(action);
            case "earliest-shortest-to-most-free-dc" -> executeEarliestShortestCloudletToMostFreeDcAction();
            case "earliest-shortest-to-nearest-dc" -> executeEarliestShortestCloudletToNearestDcAction();
            case "random-to-most-free-dc" -> executeRandomCloudletToMostFreeDcAction();
            case "earliest-most-critical-to-nearest-dc" -> executeEarliestMostCriticalCloudletToNearestDcAction();
            default -> throw new IllegalArgumentException("Cloudlet-to-DC Assignment Policy"
                    + settings.getCloudletToDcAssignmentPolicy() + " was not found!");
        };
    }

    private List<DatacenterWithType> getOrderedDatacentersForCloudlet(Cloudlet cloudlet) {
        // Step 1: Get the datacenter list
        List<Datacenter> datacenterList =
                cloudSimProxy.getSimulation().getCis().getDatacenterList();

        // Step 2: Get the location index from the cloudlet
        int loc = ((CloudletWithLocation) cloudlet).getLocation();

        // Step 3: Get the datacenter corresponding to the location
        DatacenterWithType dc = (DatacenterWithType) datacenterList.get(loc);

        // Step 4: Initialize the result list with the selected datacenter
        List<DatacenterWithType> resultList = new ArrayList<>();
        resultList.add(dc); // Assuming the primary datacenter is of
                            // type "edge" by default

        // Step 5: Get the connected datacenters from the "connectTo" array
        List<Integer> connectToArray = dc.getConnectTo();
        LOGGER.info("dc {} has connectTo {}", dc.getId(), dc.getConnectTo().toString());
        // datacenter indices

        List<DatacenterWithType> connectedDatacenters = new ArrayList<>();

        for (int i = 0; i < connectToArray.size(); i++) {
            DatacenterWithType connectedDatacenter = (DatacenterWithType) datacenterList.get(i);
            connectedDatacenters.add(connectedDatacenter);
        }

        // Step 6: Sort the connected datacenters - "edge" ones first, then "cloud"
        connectedDatacenters = connectedDatacenters.stream()
                .sorted(Comparator.comparing(DatacenterWithType::getType,
                        Comparator.reverseOrder())) // "edge" before "cloud"
                .collect(Collectors.toList());

        // Step 7: Add all connected datacenters to the result list
        resultList.addAll(connectedDatacenters);

        return resultList;
    }

    private double[] executeEarliestShortestCloudletToNearestDcAction() {
        final double targetTime = cloudSimProxy.calculateTargetTime();
        final List<Cloudlet> jobsWaitingList =
                cloudSimProxy.getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);
        // final List<Datacenter> datacenterList =
        // cloudSimProxy.getSimulation().getCis().getDatacenterList();
        // final Map<Datacenter, Long> dcFreePesMap = datacenterList.stream().collect(
        // Collectors.toMap(datacenter -> datacenter, datacenter -> datacenter.getHostList()
        // .stream().flatMap(host -> host.getVmList().stream()).mapToLong(vm -> {
        // long usedPes = vm.getCloudletScheduler().getCloudletList().stream()
        // .mapToLong(cloudlet -> cloudlet.getPesNumber()).sum();
        // return vm.getPesNumber() - usedPes;
        // }).sum()));
        int jobsPlaced = 0;
        int quality = 0;

        while (!jobsToProcessList.isEmpty()) {
            // Step 1: Find cloudlets with the earliest deadline
            double earliestDeadline = jobsToProcessList.stream()
                    .mapToDouble(
                            c -> c.getSubmissionDelay() + ((CloudletWithLocation) c).getDeadline())
                    .min().orElse(Double.MAX_VALUE);

            // Filter cloudlets with the earliest deadline
            List<Cloudlet> earliestDeadlineCloudlets = jobsToProcessList.stream()
                    .filter(c -> (c.getSubmissionDelay()
                            + ((CloudletWithLocation) c).getDeadline()) == earliestDeadline)
                    .collect(Collectors.toList());

            // From these, select the shortest one(s)
            long shortestLength = earliestDeadlineCloudlets.stream().mapToLong(Cloudlet::getLength)
                    .min().orElseThrow();

            Cloudlet selectedCloudlet = earliestDeadlineCloudlets.stream()
                    .filter(c -> c.getLength() == shortestLength).findFirst().orElseThrow();

            List<DatacenterWithType> sortedDcs = getOrderedDatacentersForCloudlet(selectedCloudlet);

            Vm targetVm = Vm.NULL;
            for (DatacenterWithType datacenter : sortedDcs) {
                targetVm = getMostFreeVmOfDcForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    cloudSimProxy.getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    jobsPlaced++;
                    quality +=
                            calculateQualityOfPlacement((int) datacenter.getId(), selectedCloudlet);

                    // Update the free PEs in dcFreePesMap
                    // long updatedFreePes =
                    // dcFreePesMap.get(datacenter) - selectedCloudlet.getPesNumber();
                    // dcFreePesMap.put(datacenter, updatedFreePes);
                    break; // Stop searching once a suitable VM is found
                }
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double[] executeEarliestMostCriticalCloudletToNearestDcAction() {
        final double targetTime = cloudSimProxy.calculateTargetTime();
        final List<Cloudlet> jobsWaitingList =
                cloudSimProxy.getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);
        // final List<Datacenter> datacenterList =
        // cloudSimProxy.getSimulation().getCis().getDatacenterList();
        // final Map<Datacenter, Long> dcFreePesMap = datacenterList.stream().collect(
        // Collectors.toMap(datacenter -> datacenter, datacenter -> datacenter.getHostList()
        // .stream().flatMap(host -> host.getVmList().stream()).mapToLong(vm -> {
        // long usedPes = vm.getCloudletScheduler().getCloudletList().stream()
        // .mapToLong(cloudlet -> cloudlet.getPesNumber()).sum();
        // return vm.getPesNumber() - usedPes;
        // }).sum()));
        int jobsPlaced = 0;
        int quality = 0;

        while (!jobsToProcessList.isEmpty()) {
            // Step 1: Find cloudlets with the earliest deadline
            double earliestDeadline = jobsToProcessList.stream()
                    .mapToDouble(
                            c -> c.getSubmissionDelay() + ((CloudletWithLocation) c).getDeadline())
                    .min().orElse(Double.MAX_VALUE);

            // Filter cloudlets with the earliest deadline
            List<Cloudlet> earliestDeadlineCloudlets = jobsToProcessList.stream()
                    .filter(c -> (c.getSubmissionDelay()
                            + ((CloudletWithLocation) c).getDeadline()) == earliestDeadline)
                    .collect(Collectors.toList());

            // From these, select the shortest one(s)
            int mostCritical = earliestDeadlineCloudlets.stream()
                    .mapToInt(c -> ((CloudletWithLocation) c).getDelaySensitivity()).max()
                    .orElseThrow();

            CloudletWithLocation selectedCloudlet = (CloudletWithLocation) earliestDeadlineCloudlets
                    .stream()
                    .filter(c -> ((CloudletWithLocation) c).getDelaySensitivity() == mostCritical)
                    .findFirst().orElseThrow();

            List<DatacenterWithType> sortedDcs = getOrderedDatacentersForCloudlet(selectedCloudlet);

            Vm targetVm = Vm.NULL;
            for (DatacenterWithType datacenter : sortedDcs) {
                targetVm = getMostFreeVmOfDcForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    cloudSimProxy.getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    jobsPlaced++;
                    quality +=
                            calculateQualityOfPlacement((int) datacenter.getId(), selectedCloudlet);

                    // Update the free PEs in dcFreePesMap
                    // long updatedFreePes =
                    // dcFreePesMap.get(datacenter) - selectedCloudlet.getPesNumber();
                    // dcFreePesMap.put(datacenter, updatedFreePes);
                    break; // Stop searching once a suitable VM is found
                }
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double[] executeEarliestShortestCloudletToMostFreeDcAction() {
        final double targetTime = cloudSimProxy.calculateTargetTime();
        final List<Cloudlet> jobsWaitingList =
                cloudSimProxy.getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);
        final List<Datacenter> datacenterList =
                cloudSimProxy.getSimulation().getCis().getDatacenterList();
        final Map<Datacenter, Long> dcFreePesMap = datacenterList.stream().collect(
                Collectors.toMap(datacenter -> datacenter, datacenter -> datacenter.getHostList()
                        .stream().flatMap(host -> host.getVmList().stream()).mapToLong(vm -> {
                            long usedPes = vm.getCloudletScheduler().getCloudletList().stream()
                                    .mapToLong(cloudlet -> cloudlet.getPesNumber()).sum();
                            return vm.getPesNumber() - usedPes;
                        }).sum()));
        int jobsPlaced = 0;
        int quality = 0;

        while (!jobsToProcessList.isEmpty()) {
            // Step 1: Find cloudlets with the earliest deadline
            double earliestDeadline = jobsToProcessList.stream()
                    .mapToDouble(
                            c -> c.getSubmissionDelay() + ((CloudletWithLocation) c).getDeadline())
                    .min().orElse(Double.MAX_VALUE);

            // Filter cloudlets with the earliest deadline
            List<Cloudlet> earliestDeadlineCloudlets = jobsToProcessList.stream()
                    .filter(c -> (c.getSubmissionDelay()
                            + ((CloudletWithLocation) c).getDeadline()) == earliestDeadline)
                    .collect(Collectors.toList());

            // From these, select the shortest one(s)
            long shortestLength = earliestDeadlineCloudlets.stream().mapToLong(Cloudlet::getLength)
                    .min().orElseThrow();

            Cloudlet selectedCloudlet = earliestDeadlineCloudlets.stream()
                    .filter(c -> c.getLength() == shortestLength).findFirst().orElseThrow();

            // // Step 2: Find the datacenter with the most free PEs
            // Datacenter targetDatacenter =
            // dcFreePesMap.entrySet().stream().max(Map.Entry.comparingByValue())
            // .map(Map.Entry::getKey).orElse(Datacenter.NULL);

            // if (targetDatacenter == Datacenter.NULL) {
            // // No datacenters are available
            // jobsToProcessList.remove(selectedCloudlet);
            // continue;
            // }

            // Step 3: Traverse datacenters in descending order of free PEs
            List<Map.Entry<Datacenter, Long>> sortedDcs = dcFreePesMap.entrySet().stream()
                    .sorted(Map.Entry.<Datacenter, Long>comparingByValue().reversed())
                    .collect(Collectors.toList());

            Vm targetVm = Vm.NULL;
            for (Iterator<Map.Entry<Datacenter, Long>> it = sortedDcs.iterator(); it.hasNext();) {
                Datacenter datacenter = it.next().getKey();
                targetVm = getMostFreeVmOfDcForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    cloudSimProxy.getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    jobsPlaced++;
                    quality +=
                            calculateQualityOfPlacement((int) datacenter.getId(), selectedCloudlet);

                    // Update the free PEs in dcFreePesMap
                    long updatedFreePes =
                            dcFreePesMap.get(datacenter) - selectedCloudlet.getPesNumber();
                    dcFreePesMap.put(datacenter, updatedFreePes);
                    break; // Stop searching once a suitable VM is found
                }
                it.remove(); // Remove datacenter from the list for this cloudlet
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double[] executeRandomCloudletToMostFreeDcAction() {
        return new double[] {0.0};
    }

    // this action is if the agent performs cloudlet to DC mapping
    private double[] executeRlCloudletToDcAction(final int[] action) {
        // final int[] actionSucess = new int[jobsWaitingThisTimestep];
        // LOGGER.info("VMs running: {}",
        // cloudSimProxy.getBroker().getVmExecList().size());
        // alternative way is to have the agent return -1 for jobs not waiting, so you
        // return the
        // index - 1, where index is the index of the first -1 in the array
        final double targetTime = cloudSimProxy.calculateTargetTime();
        final List<Cloudlet> jobsToSubmit = cloudSimProxy.getJobsToSubmitAtThisTimestep(targetTime);
        final int jobsWaiting = jobsToSubmit.size();

        int jobsPlaced = 0;
        double quality = 0.0;
        for (int i = 0; i < jobsWaiting; i++) {
            final CloudletWithLocation job = (CloudletWithLocation) jobsToSubmit.get(i);
            final int dcId = action[i] + 1;
            LOGGER.info("Action[{}]: {}", i, dcId);
            if (dcId == 1) {
                LOGGER.info("No action for Cloudlet {}", job.getId());
                continue;
            }
            final Vm vm = getMostFreeVmOfDcForCloudlet(dcId, job);
            if (vm == Vm.NULL) {
                // This should never happen because the agent should not return an action that
                // is
                // not possible. The agent knows the free cores of each DC.
                LOGGER.warn("No available VM for job {} in DC {}", job.getId(), dcId);
                continue;
            }
            LOGGER.info("Binding Cloudlet {} to VM{}/H{}/DC{}", job.getId(), vm.getId(),
                    vm.getHost().getId(), dcId);
            cloudSimProxy.getBroker().bindCloudletToVm(job, vm);
            // or simply job.setVm(vm);
            LOGGER.info("Cloudlet {} getVm {} ", job.getId(), job.getVm().getId());
            quality += calculateQualityOfPlacement(dcId, job);
            jobsPlaced++;
        }

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaiting);
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsToSubmit);
        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double calculateJobsPlacedRatio(final int jobsPlaced, final int jobsWaiting) {
        if (jobsWaiting == 0) {
            return 0.0;
        }
        return (double) jobsPlaced / jobsWaiting;
    }

    private double calculateQualityRatio(final double quality, final int jobsPlaced) {
        if (jobsPlaced == 0) {
            return 0.0;
        }
        return quality / jobsPlaced;
    }

    private double calculateDeadlineViolationRatio(final List<Cloudlet> jobsWaiting) {
        if (jobsWaiting.size() == 0) {
            return 0;
        }
        final double targetTime = cloudSimProxy.calculateTargetTime();
        final long deadlineViolations = jobsWaiting.stream()
                .filter(job -> targetTime > job.getSubmissionDelay()
                        + ((CloudletWithLocation) job).getDeadline() && job.getVm() == Vm.NULL)
                .count();
        return (double) deadlineViolations / jobsWaiting.size();
    }

    // This is for the VM management action
    // private int[] executeCustomVmManagementAction(final int[] action) {
    // returns [hostId, coresChanged]

    // final boolean isValid;

    // LOGGER.info("{}: Timestep: {}, Action: [{}, {}, {}, {}]", clock(),
    // currentStep,
    // action[0],
    // action[1], action[2], action[3]);

    // [action, hostId, vmId, type]
    // action = {0: do nothing, 1: create vm, 2: destroy vm}
    // id = {hostId to place new vm (when action = 1), vmId to terminate (when
    // action = 2)
    // type = {0: small, 1: medium, 2: large} (relevant only when action = 1)

    // if (action[0] == 1) {
    // final int hostId = action[1];
    // final int vmTypeIndex = action[3];
    // final int vmCores =
    // cloudSimProxy.getVmCoreCountByType(settings.VM_TYPES[vmTypeIndex]);
    // isValid = addNewVm(settings.VM_TYPES[vmTypeIndex], hostId);
    // if (!isValid) {
    // return new int[] {-1, 0};
    // }
    // return new int[] {hostId, vmCores};
    // }

    // else if (action[0] == 2) {
    // final int vmIndex = action[2];
    // List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
    // Vm vm = vmList.get(vmIndex);
    // int hostId = (int) vm.getHost().getId();
    // int vmCores = (int) vm.getPesNumber();
    // isValid = removeVm(vmIndex);
    // if (!isValid) {
    // return new int[] {-1, 0};
    // }
    // return new int[] {hostId, vmCores};
    // }
    // }

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
    // settings.getDatacenterHostsCnt());

    // vmTypeIndex = (int) continuousToDiscrete(
    // action[1],
    // settings.VM_TYPES.length);

    // debug("Translated action[0] = " + id);
    // debug("Will try to create a new vm at host with id = "
    // + id + " of type " + settings.VM_TYPES[vmTypeIndex]);
    // isValid = addNewVm(settings.VM_TYPES[vmTypeIndex], id);
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
            LOGGER.warn("Adding a VM of type {} to host {} is invalid. Ignoring", type, hostId);
            return false;
        }
        return true;
    }

    // private void updateMetrics() {
    // metricsStorage.setDatacenterMetrics(collectDatacenterMetrics());
    // metricsStorage.setHostMetrics(collectHostMetrics());
    // metricsStorage.setVmMetrics(collectVmMetrics());
    // metricsStorage.setJobMetrics(collectJobMetrics());
    // }

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
                ? cloudSimProxy.getNotYetRunningJobsCount() / (double) arrivedJobsCount
                : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        return 0.0;
        // return ((double) cloudSimProxy.getAllocatedCores()) /
        // settings.getTotalHostCores();
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

    /**
     * Extracts a subarray from the given matrix by selecting specific columns.
     *
     * @param matrix The original 2D array from which the subarray will be extracted.
     * @param columnIndices An array of column indices to be included in the subarray.
     * @return A 2D array containing the selected columns from the original matrix.
     */
    // private double[][] getVertSubarray(final double[][] matrix, final int[]
    // columnIndices) {
    // int numRows = matrix.length;
    // int numCols = columnIndices.length;
    // double[][] result = new double[numRows][numCols];
    // for (int i = 0; i < numRows; i++) {
    // for (int j = 0; j < numCols; j++) {
    // result[i][j] = matrix[i][columnIndices[j]];
    // }
    // }
    // return result;
    // }

    private int[] getInfrastructureObservation() {
        switch (settings.getStateActionSpaceType()) {
            case "dcid-dctype-freevmpes-per-host":
                return getInfraObsDcIdDcTypeFreeVmPesPerHost();
            default:
                throw new IllegalArgumentException(
                        "Unexpected value: " + settings.getStateActionSpaceType());
        }
    }

    private int[] getJobsWaitingObservation() {
        final int[] jobWaitObs = cloudSimProxy.getJobsWaitingObservation();
        // 4 should not be hardcoded. Same here for python side.
        final int jobsWaiting = jobWaitObs.length / 4;
        LOGGER.info("Jobs waiting: {}", jobsWaiting);
        LOGGER.info("JobWaitObs: {}", Arrays.toString(jobWaitObs));
        // for (int i = 0; i < jobsWaiting; i++) {
        // LOGGER.info("Waiting job: {}, Cores: {}, Location: {}", i, coreLocObs[2 * i],
        // coreLocObs[2 * i + 1]);
        // }
        return jobWaitObs;
    }

    // private int getTotalJobCoresWaitingObservation() {
    // final int jobCoresWaiting = cloudSimProxy.calculateJobCoresWaiting();
    // final int largeVmPes = settings.getSmallVmPes() *
    // settings.getLargeVmMultiplier();
    // // Do not allow the observation to be larger than the number of cores in the
    // // large VM
    // return Math.min(jobCoresWaiting, largeVmPes);
    // }

    private int getMaxVmPesAcrossAllDc() {
        final List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        return vmList.stream().map(Vm::getPesNumber).reduce(0L, Math::max).intValue();
    }

    /**
     * Retrieves the total number of free VM cores per host in the infrastructure.
     * <p>
     * This method assumes that the trace file contains cloudlets, and VMs have already been opened
     * to fit inside all hosts. Therefore, the free cores of interest are the free cores of the VMs.
     * It also assumes that each host has only one VM that is as large as the host. Consequently,
     * the method counts the free cores of the VMs.
     * <p>
     * If the trace file contains VMs, no VMs should be opened, and the free cores of the hosts
     * should be counted instead.
     * <p>
     * The method returns an array where each pair of elements represents a datacenter ID and the
     * corresponding number of free cores in that datacenter.
     * 
     * @return an array of integers where each pair of elements represents a datacenter ID and the
     *         corresponding number of free cores in that datacenter.
     */
    private int[] getInfraObsDcIdDcTypeFreeVmPesPerHost() {
        final int totalHosts = getTotalHosts();
        final int[] infrastructureObservation = new int[3 * totalHosts];
        List<Datacenter> datacenterList =
                cloudSimProxy.getSimulation().getCis().getDatacenterList();
        int currentIndex = 0;
        for (Datacenter dc : datacenterList) {
            for (Host host : dc.getHostList()) {
                int freePes = 0;
                final List<Vm> vmList = host.getVmList();
                // - 1 because dc ids start from 2, Actions start with 0 but 0 means no dc, so
                // we send 1 that means dc with id 2.
                // We do the opposite (add 1) when we get the action
                infrastructureObservation[currentIndex++] = (int) dc.getId() - 1;
                infrastructureObservation[currentIndex++] =
                        getDcTypeIdFromStr(((DatacenterWithType) dc).getType());
                for (Vm vm : vmList) {
                    List<Cloudlet> cloudletList = vm.getCloudletScheduler().getCloudletList();
                    long usedPes = cloudletList.stream().mapToLong(Cloudlet::getPesNumber).sum();
                    freePes += vm.getPesNumber() - usedPes;
                    LOGGER.info("Writing {} in the observation", freePes);
                    LOGGER.info("vm.getFreePesNumber(): {}", vm.getFreePesNumber());
                    // freePes += vm.getFreePesNumber();
                }
                infrastructureObservation[currentIndex++] = freePes;
            }
        }
        LOGGER.info("InfrObs: {}", Arrays.toString(infrastructureObservation));
        return infrastructureObservation;
    }

    private int getDcTypeIdFromStr(final String dcType) {
        return switch (dcType) {
            case "cloud" -> 0;
            case "edge" -> 1;
            case "micro" -> 2;
            default -> throw new IllegalArgumentException("Unexpected DC type: " + dcType);
        };
    }

    private int getTotalHosts() {
        int totalHosts = 0;
        List<Datacenter> datacenterList =
                cloudSimProxy.getSimulation().getCis().getDatacenterList();
        for (Datacenter datacenter : datacenterList) {
            List<Host> hostList = datacenter.getHostList();
            totalHosts += hostList.size();
        }
        return totalHosts;
    }

    // private int[] getInfrastructureObservationAsTreeArray() {
    // final int hostsNum = settings.getHostsCount();
    // final int vmsNum = getRunningVmsCount().intValue();
    // final int jobsNum = getRunningCloudletsCount().intValue();
    // final int[] treeArray = new int[2 + 2 * hostsNum + 2 * vmsNum + 2 * jobsNum];

    // final int totalDatacenterCores = (int) settings.getDatacenterCores();
    // final List<Host> hostList = cloudSimProxy.getDatacenter().getHostList();
    // treeArray[0] = totalDatacenterCores;
    // treeArray[1] = hostsNum;
    // int currentIndex = 2;
    // for (int i = 0; i < hostsNum; i++) {
    // final Host host = hostList.get(i);
    // final List<Vm> vmList = host.getVmList();
    // treeArray[currentIndex++] = (int) host.getPesNumber();
    // treeArray[currentIndex++] = vmList.size();
    // for (int j = 0; j < vmList.size(); j++) {
    // final Vm vm = vmList.get(j);
    // final List<Cloudlet> jobList = vm.getCloudletScheduler().getCloudletList();
    // treeArray[currentIndex++] = (int) vm.getPesNumber();
    // treeArray[currentIndex++] = jobList.size();
    // for (int k = 0; k < jobList.size(); k++) {
    // final Cloudlet cloudlet = jobList.get(k);
    // treeArray[currentIndex++] = (int) cloudlet.getPesNumber();
    // treeArray[currentIndex++] = 0; // jobs do not have children
    // }
    // }
    // }
    // // System.out.print(clock() + " TreeArray in wrappedSimulation: ");
    // // System.out.println(Arrays.deepToString(treeArray));

    // return treeArray;
    // }

    // private double[][] getInfrastructureObservationAs2dArray() {
    // // here we get some vertical subarrays of the metrics. The whole array of
    // // metrics are only
    // // used to pass the info to python and print to csv files then.
    // // Vertical because rows in the arrays represent the hosts, vms, jobs etc.
    // So,
    // // we take a specific subset of features (columns) for every host, vm or job.

    // final double[] datacenterMetrics = new double[]
    // {metricsStorage.getDatacenterMetrics()[2]};
    // final double[][] hostMetrics =
    // getVertSubarray(metricsStorage.getHostMetrics(), new int[] {3, 5, 7, 8});
    // final double[][] vmMetrics = getVertSubarray(metricsStorage.getVmMetrics(),
    // new int[] {5});
    // final double[][] jobMetrics =
    // getVertSubarray(metricsStorage.getJobMetrics(), new int[] {5});
    // final double[][] observation = new
    // double[observationArrayRows][observationArrayColumns];
    // /*
    // * if you want to support 1-10 hosts, then below when assigning new values
    // after loops for
    // * currentRow put maxHosts instead of hostsCount
    // */
    // int currentRow = 0;

    // for (int j = 0; j < datacenterMetrics.length; j++) {
    // observation[currentRow][j] = datacenterMetrics[j];
    // }
    // currentRow++;

    // for (int i = 0; i < hostMetrics.length; i++) {
    // for (int j = 0; j < hostMetrics[i].length; j++) {
    // observation[currentRow][j] = hostMetrics[i][j];
    // }
    // currentRow++;
    // }

    // currentRow = 1 + maxHosts;

    // for (int i = 0; i < vmMetrics.length; i++) {
    // for (int j = 0; j < vmMetrics[i].length; j++) {
    // observation[currentRow][j] = vmMetrics[i][j];
    // }
    // currentRow++;
    // }

    // currentRow = 1 + maxHosts + maxVms;

    // for (int i = 0; i < jobMetrics.length; i++) {
    // for (int j = 0; j < jobMetrics[i].length; j++) {
    // observation[currentRow][j] = jobMetrics[i][j];
    // }
    // currentRow++;
    // }

    // return observation;
    // }

    // private double calculateJobPlacementRatioReward() {
    // final int jobsWaitingThisTimestep = ;
    // if (jobsWaitingThisTimestep == 0) {
    // return 0.0;
    // }
    // final double jobsPlacedCoef = settings.getRewardJobsPlacedCoef();
    // final double jobsPlacedReward = (double) jobsPlaced /
    // jobsWaitingThisTimestep;
    // return jobsPlacedCoef * jobsPlacedReward;
    // }

    private double calculateReward(final double jobsPlacedRatio, final double qualityRatio,
            final double deadlineViolationRatio) {
        /*
         * reward is the negative cost of running the infrastructure minus any penalties from jobs
         * waiting in the queue minus penalty if action was invalid
         */

        final double jobsPlacedCoef = settings.getRewardJobsPlacedCoef();
        final double qualityCoef = settings.getRewardQualityCoef();
        final double deadlineViolationCoef = settings.getRewardDeadlineViolationCoef();

        final double reward = jobsPlacedCoef * jobsPlacedRatio + qualityCoef * qualityRatio
                - deadlineViolationCoef * deadlineViolationRatio;

        LOGGER.info("totalReward: {}", reward);
        LOGGER.info("jobsPlacedReward: {}", jobsPlacedCoef * jobsPlacedRatio);
        LOGGER.info("qualityReward: {}", qualityCoef * qualityRatio);
        LOGGER.info("deadlineMissReward: {}", deadlineViolationCoef * deadlineViolationRatio);

        return reward;
    }

    public SimulationSettings getSettings() {
        return settings;
    }

    // private double[] calculateReward(final boolean isValid) {
    // double[] rewards = new double[5];
    // /*
    // * reward is the negative cost of running the infrastructure minus any
    // penalties from jobs
    // * waiting in the queue minus penalty if action was invalid
    // */

    // final double jobWaitCoef = settings.getRewardJobWaitCoef();
    // final double runningVmCoresCoef = settings.getRewardRunningVmCoresCoef();
    // final double unutilizedVmCoresCoef =
    // settings.getRewardUnutilizedVmCoresCoef();
    // final double invalidCoef = settings.getRewardInvalidCoef();

    // final double jobWaitReward = -jobWaitCoef * getWaitingJobsRatio();
    // final double runningVmCoresReward = -runningVmCoresCoef *
    // getHostCoresAllocatedToVmsRatio();
    // final double unutilizedVmCoresReward = -unutilizedVmCoresCoef *
    // getUnutilizedVmCoreRatio();
    // final double invalidReward = -invalidCoef * (isValid ? 0 : 1);

    // double totalReward = 0;
    // if (settings.getVmAllocationPolicy().equals("rule-based")) {
    // totalReward = jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward;
    // } else if (settings.getVmAllocationPolicy().equals("rl")) {
    // totalReward =
    // jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward +
    // invalidReward;
    // } else {
    // LOGGER.error(identifier + ": Invalid VM allocation policy");
    // }

    // LOGGER.info("totalReward: " + totalReward);
    // LOGGER.info("jobWaitReward: " + jobWaitReward);
    // LOGGER.info("runningVmCoresReward: " + runningVmCoresReward);
    // LOGGER.info("unutilizedVmCoresReward: " + unutilizedVmCoresReward);
    // LOGGER.info("invalidReward: " + invalidReward);

    // rewards[0] = totalReward;
    // rewards[1] = jobWaitReward;
    // rewards[2] = runningVmCoresReward;
    // rewards[3] = unutilizedVmCoresReward;
    // rewards[4] = invalidReward;

    // if (!isValid) {
    // LOGGER.debug("Penalty given to the agent because the selected action was not
    // possible");
    // }
    // return rewards;
    // }

    public int getCurrentStep() {
        return currentStep;
    }

    // private void resetEpRunningVmsCountMax() {
    // epRunningVmsCountMax = 0;
    // }

    // private void resetEpWaitingJobsCountMax() {
    // epWaitingJobsCountMax = 0;
    // }

    // private long getEpRunningVmsCountMax() {
    // return epRunningVmsCountMax;
    // }

    // private long getEpWaitingJobsCountMax() {
    // return epWaitingJobsCountMax;
    // }

    // private void updateEpWaitingJobsCountMax(long waitingJobsCount) {
    // if (waitingJobsCount > epWaitingJobsCountMax) {
    // epWaitingJobsCountMax = waitingJobsCount;
    // }
    // }

    // private void updateEpRunningVmsCountMax(long runningVms) {
    // if (runningVms > epRunningVmsCountMax) {
    // epRunningVmsCountMax = runningVms;
    // }
    // }

    public String getIdentifier() {
        return identifier;
    }

    public double clock() {
        return cloudSimProxy.clock();
    }
}

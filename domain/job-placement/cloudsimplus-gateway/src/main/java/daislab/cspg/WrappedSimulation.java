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


public class WrappedSimulation extends WrappedSimulationBase {

    // Concrete settings reference for domain-specific access
    private final SimulationSettings simSettings;

    // RL-level episode tracking (RL concepts; not present in CloudSimProxy)
    private int bestEpisodeReward;
    private int currentEpisodeReward;
    private double lastReward = 0.0;

    // Action-phase placement counter: set by action methods, consumed in step info
    private int jobsPlacedThisTimestep;

    public WrappedSimulation(final String identifier, final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        super(identifier, settings, jobs);
        this.simSettings = (SimulationSettings) settings;
        bestEpisodeReward = -Integer.MAX_VALUE;
    }

    // ============== Abstract method implementations ==============

    @Override
    protected ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets) {
        return new CloudSimProxy(simSettings, cloudlets);
    }

    @Override
    protected int[] extractInfrastructureObservation() {
        switch (simSettings.getStateSpaceType()) {
            case "dcid-dctype-freevmpes-per-host":
                return getInfraObsDcIdDcTypeFreeVmPesPerHost();
            default:
                throw new IllegalArgumentException(
                        "Unexpected value: " + simSettings.getStateSpaceType());
        }
    }

    @Override
    protected int[] extractSecondaryObservation() {
        return getJobsWaitingObservation();
    }

    // ============== Override reset() to reset episode counters ==============

    @Override
    public SimulationResetResult reset(final long seed) {
        this.currentEpisodeReward = 0;
        return super.reset(seed);
    }

    // ============== Override step() for jp-specific action/reward flow ==============

    @Override
    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();
        currentStep++;
        LOGGER.info("Step {} starting", currentStep);
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;

        final int driftAt = simSettings.getDriftAtStep();
        if (driftAt > 0 && currentStep == driftAt) {
            LOGGER.info("Drift event at step {}: failing DC index {}", currentStep,
                    simSettings.getDriftDcIndex());
            proxy.injectDrift(simSettings.getDriftDcIndex());
        }

        final double[] ratios = executeCustomCloudletToDcAction(action);

        final double targetTime = proxy.calculateTargetTime();
        final int jobsWaiting = proxy.getJobsToSubmitAtThisTimestep(targetTime).size();

        proxy.runOneTimestep();

        boolean terminated = !proxy.isRunning();
        boolean truncated = !terminated && (currentStep >= simSettings.getMaxEpisodeLength());

        double reward = calculateReward(ratios[0], ratios[1], ratios[2]);
        this.lastReward = reward;
        this.currentEpisodeReward += reward;

        LOGGER.info("Step {} finished", currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", proxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    proxy.getBroker().getCloudletFinishedList().size(),
                    initialJobsDescriptors.size());
            if (currentEpisodeReward > bestEpisodeReward) {
                bestEpisodeReward = currentEpisodeReward;
                LOGGER.info("New best episode reward: {}", bestEpisodeReward);
            }
        }

        SimulationStepInfo info = new SimulationStepInfo(jobsWaiting, this.jobsPlacedThisTimestep,
                ratios[0], ratios[1], ratios[2], proxy.getFinishedJobsWaitTimeLastTimestep());

        final Observation observation =
                buildObservation(extractInfrastructureObservation(), extractSecondaryObservation());
        return new SimulationStepResult(observation, reward, terminated, truncated, info);
    }

    // Casts the inherited cloudSimProxy field to the concrete type used by jp
    private CloudSimProxy proxy() {
        return (CloudSimProxy) cloudSimProxy;
    }

    private Vm getMostFreeVmOfDcForCloudlet(final int targetDcId, final Cloudlet cloudlet) {
        long maxExpectedFreePes = 0;
        Vm mostFreeVm = Vm.NULL;
        final double targetTime = proxy().calculateTargetTime();
        List<Vm> vmList = proxy().getBroker().getVmExecList();
        List<Cloudlet> cloudletList = proxy().getJobsToSubmitAtThisTimestep(targetTime);

        Map<Vm, Long> expectedToUseVmPesMap =
                vmList.stream().collect(Collectors.toMap(vm -> vm, vm -> cloudletList.stream()
                        .filter(c -> c.getVm() == vm).mapToLong(Cloudlet::getPesNumber).sum()));

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

            if (dcId == targetDcId && vm.isSuitableForCloudlet(cloudlet)
                    && expectedFreePes >= cloudlet.getPesNumber()) {
                if (expectedFreePes > maxExpectedFreePes) {
                    maxExpectedFreePes = expectedFreePes;
                    mostFreeVm = vm;
                }
            }
        }

        LOGGER.debug("{}: Selecting VM {} for cloudlet {} with {} expected free cores", clock(),
                mostFreeVm.getId(), cloudlet.getId(), maxExpectedFreePes);
        return mostFreeVm;
    }

    private double calculateQualityOfPlacement(final int dcId, final Cloudlet job) {
        final String datacenterType =
                ((DatacenterWithType) proxy().getDatacenterById(dcId)).getType();
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
        return switch (simSettings.getCloudletToDcMapping()) {
            case "rl" -> executeRlCloudletToDcAction(action);
            case "earliest-shortest-to-most-free-dc" -> executeEarliestShortestCloudletToMostFreeDcAction();
            case "earliest-shortest-to-nearest-dc" -> executeEarliestShortestCloudletToNearestDcAction();
            case "earliest-most-critical-to-nearest-dc" -> executeEarliestMostCriticalCloudletToNearestDcAction();
            default -> throw new IllegalArgumentException("Unknown cloudlet_to_dc_mapping: "
                    + simSettings.getCloudletToDcMapping());
        };
    }

    private Vm selectVmForCloudlet(final int dcId, final Cloudlet cloudlet) {
        return switch (simSettings.getCloudletToVmMapping()) {
            case "most-free-pes" -> getMostFreeVmOfDcForCloudlet(dcId, cloudlet);
            default -> throw new IllegalArgumentException("Unknown cloudlet_to_vm_mapping: "
                    + simSettings.getCloudletToVmMapping());
        };
    }

    private List<DatacenterWithType> getOrderedDatacentersForCloudlet(Cloudlet cloudlet) {
        // Step 1: Get the datacenter list
        List<Datacenter> datacenterList = proxy().getSimulation().getCis().getDatacenterList();

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
        final double targetTime = proxy().calculateTargetTime();
        final List<Cloudlet> jobsWaitingList = proxy().getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);

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
                targetVm = selectVmForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    proxy().getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    jobsPlaced++;
                    quality +=
                            calculateQualityOfPlacement((int) datacenter.getId(), selectedCloudlet);

                    break; // Stop searching once a suitable VM is found
                }
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

        this.jobsPlacedThisTimestep = jobsPlaced;

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double[] executeEarliestMostCriticalCloudletToNearestDcAction() {
        final double targetTime = proxy().calculateTargetTime();
        final List<Cloudlet> jobsWaitingList = proxy().getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);

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
                targetVm = selectVmForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    proxy().getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    jobsPlaced++;
                    quality +=
                            calculateQualityOfPlacement((int) datacenter.getId(), selectedCloudlet);

                    break; // Stop searching once a suitable VM is found
                }
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

        this.jobsPlacedThisTimestep = jobsPlaced;

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    private double[] executeEarliestShortestCloudletToMostFreeDcAction() {
        final double targetTime = proxy().calculateTargetTime();
        final List<Cloudlet> jobsWaitingList = proxy().getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);
        final List<Datacenter> datacenterList =
                proxy().getSimulation().getCis().getDatacenterList();
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

            // Step 2: Traverse datacenters in descending order of free PEs
            List<Map.Entry<Datacenter, Long>> sortedDcs = dcFreePesMap.entrySet().stream()
                    .sorted(Map.Entry.<Datacenter, Long>comparingByValue().reversed())
                    .collect(Collectors.toList());

            Vm targetVm = Vm.NULL;
            for (Iterator<Map.Entry<Datacenter, Long>> it = sortedDcs.iterator(); it.hasNext();) {
                Datacenter datacenter = it.next().getKey();
                targetVm = selectVmForCloudlet((int) datacenter.getId(), selectedCloudlet);

                if (targetVm != Vm.NULL) {
                    // Found a suitable VM
                    proxy().getBroker().bindCloudletToVm(selectedCloudlet, targetVm);
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

        this.jobsPlacedThisTimestep = jobsPlaced;

        final double jobsPlacedRatio = calculateJobsPlacedRatio(jobsPlaced, jobsWaitingList.size());
        final double qualityRatio = calculateQualityRatio(quality, jobsPlaced);
        final double deadlineViolationRatio = calculateDeadlineViolationRatio(jobsWaitingList);
        LOGGER.info("jobsPlacedRatio: {}, qualityRatio: {}, deadlineViolationRatio: {}",
                jobsPlacedRatio, qualityRatio, deadlineViolationRatio);

        return new double[] {jobsPlacedRatio, qualityRatio, deadlineViolationRatio};
    }

    // this action is if the agent performs cloudlet to DC mapping
    private double[] executeRlCloudletToDcAction(final int[] action) {

        final double targetTime = proxy().calculateTargetTime();
        final List<Cloudlet> jobsToSubmit = proxy().getJobsToSubmitAtThisTimestep(targetTime);
        final int jobsWaiting = jobsToSubmit.size();

        int jobsPlaced = 0;
        double quality = 0.0;
        for (int i = 0; i < jobsWaiting; i++) {
            if (action.length <= i) {
                LOGGER.warn(
                        "More jobs waiting than actions returned by the agent. Jobs will stay in the queue. Continuing...");
                break;
            }
            final CloudletWithLocation job = (CloudletWithLocation) jobsToSubmit.get(i);
            final int dcId = action[i] + 1;
            LOGGER.info("Action[{}]: {}", i, dcId);
            if (dcId == 1) {
                LOGGER.info("No action for Cloudlet {}", job.getId());
                continue;
            }
            final Vm vm = selectVmForCloudlet(dcId, job);
            if (vm == Vm.NULL) {
                // This should never happen because the agent should not return an action that
                // is not possible. The agent knows the free cores of each DC.
                LOGGER.warn("No available VM for job {} in DC {}", job.getId(), dcId);
                continue;
            }
            LOGGER.info("Binding Cloudlet {} to VM{}/H{}/DC{}", job.getId(), vm.getId(),
                    vm.getHost().getId(), dcId);
            proxy().getBroker().bindCloudletToVm(job, vm);
            // or simply job.setVm(vm);
            LOGGER.info("Cloudlet {} getVm {} ", job.getId(), job.getVm().getId());
            quality += calculateQualityOfPlacement(dcId, job);
            jobsPlaced++;
        }

        this.jobsPlacedThisTimestep = jobsPlaced;

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
        final double targetTime = proxy().calculateTargetTime();
        final long deadlineViolations = jobsWaiting.stream()
                .filter(job -> targetTime > job.getSubmissionDelay()
                        + ((CloudletWithLocation) job).getDeadline() && job.getVm() == Vm.NULL)
                .count();
        return (double) deadlineViolations / jobsWaiting.size();
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
        List<Datacenter> datacenterList = proxy().getSimulation().getCis().getDatacenterList();
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
                }
                infrastructureObservation[currentIndex++] = freePes;
            }
        }
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
        List<Datacenter> datacenterList = proxy().getSimulation().getCis().getDatacenterList();
        for (Datacenter datacenter : datacenterList) {
            List<Host> hostList = datacenter.getHostList();
            totalHosts += hostList.size();
        }
        return totalHosts;
    }

    private double calculateReward(final double jobsPlacedRatio, final double qualityRatio,
            final double deadlineViolationRatio) {
        /*
         * reward is the negative cost of running the infrastructure minus any penalties from jobs
         * waiting in the queue minus penalty if action was invalid
         */

        final double jobsPlacedCoef = simSettings.getRewardJobsPlacedCoef();
        final double qualityCoef = simSettings.getRewardQualityCoef();
        final double deadlineViolationCoef = simSettings.getRewardDeadlineViolationCoef();

        final double reward = jobsPlacedCoef * jobsPlacedRatio + qualityCoef * qualityRatio
                - deadlineViolationCoef * deadlineViolationRatio;

        LOGGER.info("totalReward: {}", reward);
        LOGGER.info("jobsPlacedReward: {}", jobsPlacedCoef * jobsPlacedRatio);
        LOGGER.info("qualityReward: {}", qualityCoef * qualityRatio);
        LOGGER.info("deadlineMissReward: {}", deadlineViolationCoef * deadlineViolationRatio);

        return reward;
    }

    private int[] getJobsWaitingObservation() {
        final int[] jobWaitObs = proxy().getJobsWaitingObservation();
        final int jobsWaiting = jobWaitObs.length / CloudSimProxy.JOB_OBS_FEATURES;
        LOGGER.info("Jobs waiting: {}", jobsWaiting);
        LOGGER.info("JobWaitObs: {}", Arrays.toString(jobWaitObs));
        return jobWaitObs;
    }

    public SimulationSettings getSettings() {
        return simSettings;
    }

    public double getLastReward() {
        return lastReward;
    }
}

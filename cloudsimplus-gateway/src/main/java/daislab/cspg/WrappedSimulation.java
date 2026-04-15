package daislab.cspg;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.List;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WrappedSimulation {
    private final Logger LOGGER = LoggerFactory.getLogger(WrappedSimulation.class.getSimpleName());

    private final List<CloudletDescriptor> initialJobsDescriptors;

    private final String identifier;
    private final SimulationSettings settings;
    private CloudSimProxy cloudSimProxy;
    private int currentStep;

    public WrappedSimulation(final String identifier, final SimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        this.identifier = identifier;
        this.settings = settings;
        initialJobsDescriptors = jobs;
        LOGGER.info("Creating simulation: {}", identifier);
    }

    private int getJobCoresWaitingObservation() {
        final int jobCoresWaiting = cloudSimProxy.calculateJobCoresWaiting();
        final int largeVmPes = settings.getSmallVmPes() * settings.getLargeVmMultiplier();
        // Do not allow the observation to be larger than the number of cores in the
        // large VM
        return Math.min(jobCoresWaiting, largeVmPes);
    }

    public void close() {
        LOGGER.info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.terminate();
        }
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

        resetCurrentStep();

        List<Cloudlet> cloudlets = initialJobsDescriptors.stream()
                .map(CloudletDescriptor::toCloudlet).collect(Collectors.toList());
        cloudSimProxy = new CloudSimProxy(settings, cloudlets);

        SimulationStepInfo info = new SimulationStepInfo();

        Observation observation =
                new Observation(getInfrastructureObservation(), getJobCoresWaitingObservation());

        return new SimulationResetResult(observation, info);
    }

    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();
        currentStep++;

        LOGGER.info("Step {} starting", currentStep);
        int[] actionResult = switch (settings.getVmAllocationPolicy()) {
            case "rl", "fromfile" -> executeCustomAction(action);
            case "rule-based" -> {
                cloudSimProxy.executeRuleBasedAction();
                yield new int[] {0, 0}; // does not matter
            }
            default -> throw new IllegalArgumentException(
                    "Unexpected value: " + settings.getVmAllocationPolicy());
        };

        final boolean isValid = actionResult[0] != -1;

        cloudSimProxy.runOneTimestep();

        boolean terminated = !cloudSimProxy.isRunning();
        boolean truncated = !terminated && (currentStep >= settings.getMaxEpisodeLength());

        double[] rewards = calculateReward(isValid);

        LOGGER.info("Step {} finished", currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", cloudSimProxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    cloudSimProxy.getBroker().getCloudletFinishedList().size(),
                    initialJobsDescriptors.size());
        }

        final int[] treeArray = settings.getSendObservationTreeArray()
                ? getInfrastructureObservation()
                : new int[0];
        SimulationStepInfo info = new SimulationStepInfo(rewards,
                cloudSimProxy.getFinishedJobsWaitTimeLastTimestep(), getUnutilizedVmCoreRatio(),
                treeArray, actionResult[0], actionResult[1],
                settings.getSendObservationTreeArray());

        Observation observation =
                new Observation(getInfrastructureObservation(), getJobCoresWaitingObservation());

        return new SimulationStepResult(observation, rewards[0], terminated, truncated, info);
    }

    public String render() {
        return "Not Implemented yet.";
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

    private Long getRunningVmsCount() {
        return cloudSimProxy.getBroker().getVmExecList().stream().count();
    }

    private Long getRunningCloudletsCount() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();

        Long runningCloudletCount = vmList.parallelStream().map(Vm::getCloudletScheduler)
                .map(CloudletScheduler::getCloudletExecList).mapToLong(List::size).sum();
        return runningCloudletCount;
    }

    private int[] executeCustomAction(final int[] action) {
        // returns [hostId, coresChanged]

        final boolean isValid;

        LOGGER.info("{}: Timestep: {}, Action: [{}, {}, {}, {}]", clock(), currentStep, action[0],
                action[1], action[2], action[3]);

        // [action, hostId, vmId, type]
        // action = {0: do nothing, 1: create vm, 2: destroy vm}
        // id = {hostId to place new vm (when action = 1), vmId to terminate (when
        // action = 2)
        // type = {0: small, 1: medium, 2: large} (relevant only when action = 1)

        if (action[0] == 1) {
            final int hostId = action[1];
            final int vmTypeIndex = action[3];
            final int vmCores = cloudSimProxy.getVmCoreCountByType(settings.VM_TYPES[vmTypeIndex]);
            isValid = addNewVm(settings.VM_TYPES[vmTypeIndex], hostId);
            if (!isValid) {
                return new int[] {-1, 0};
            }
            return new int[] {hostId, vmCores};
        }

        else if (action[0] == 2) {
            final int vmIndex = action[2];
            List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
            Vm vm = vmList.get(vmIndex);
            int hostId = (int) vm.getHost().getId();
            int vmCores = (int) vm.getPesNumber();
            isValid = removeVm(vmIndex);
            if (!isValid) {
                return new int[] {-1, 0};
            }
            return new int[] {hostId, vmCores};
        }

        return new int[] {0, 0};
    }

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

    private double getWaitingJobsRatio() {
        final long arrivedJobsCount = cloudSimProxy.getArrivedJobsCount();

        return arrivedJobsCount > 0
                ? cloudSimProxy.getNotYetRunningJobsCount() / (double) arrivedJobsCount
                : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        return ((double) cloudSimProxy.getAllocatedCores()) / settings.getTotalHostCores();
    }

    private int[] getInfrastructureObservation() {
        final int hostsNum = settings.getHostsCount();
        final int vmsNum = getRunningVmsCount().intValue();
        final int jobsNum = getRunningCloudletsCount().intValue();
        final int[] treeArray = new int[2 * (1 + hostsNum + vmsNum + jobsNum)];

        final int totalDatacenterCores = (int) settings.getDatacenterCores();
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
        return treeArray;
    }

    private double[] calculateReward(final boolean isValid) {
        double[] rewards = new double[5];
        /*
         * reward is the negative cost of running the infrastructure minus any penalties from jobs
         * waiting in the queue minus penalty if action was invalid
         */

        final double jobWaitCoef = settings.getRewardJobWaitCoef();
        final double runningVmCoresCoef = settings.getRewardRunningVmCoresCoef();
        final double unutilizedVmCoresCoef = settings.getRewardUnutilizedVmCoresCoef();
        final double invalidCoef = settings.getRewardInvalidCoef();

        final double jobWaitReward = -jobWaitCoef * getWaitingJobsRatio();
        final double runningVmCoresReward = -runningVmCoresCoef * getHostCoresAllocatedToVmsRatio();
        final double unutilizedVmCoresReward = -unutilizedVmCoresCoef * getUnutilizedVmCoreRatio();
        final double invalidReward = -invalidCoef * (isValid ? 0 : 1);

        double totalReward = 0;
        if (settings.getVmAllocationPolicy().equals("rule-based")) {
            totalReward = jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward;
        } else if (settings.getVmAllocationPolicy().equals("rl")) {
            totalReward =
                    jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward + invalidReward;
        } else {
            LOGGER.error(identifier + ": Invalid VM allocation policy");
        }

        LOGGER.info("totalReward: " + totalReward);
        LOGGER.info("jobWaitReward: " + jobWaitReward);
        LOGGER.info("runningVmCoresReward: " + runningVmCoresReward);
        LOGGER.info("unutilizedVmCoresReward: " + unutilizedVmCoresReward);
        LOGGER.info("invalidReward: " + invalidReward);

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

    public String getIdentifier() {
        return identifier;
    }

    public double clock() {
        return cloudSimProxy.clock();
    }
}

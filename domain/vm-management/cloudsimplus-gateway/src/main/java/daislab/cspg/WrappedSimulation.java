package daislab.cspg;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.List;

/**
 * VM Management simulation using RL-driven VM lifecycle management.
 * Extends WrappedSimulationBase with tree-array infrastructure observation
 * and VM create/destroy action space.
 */
public class WrappedSimulation
        extends WrappedSimulationBase<Observation, SimulationStepInfo, SimulationStepResult, SimulationResetResult>
        implements IWrappedSimulation {

    // Concrete settings reference for domain-specific access
    private final SimulationSettings simSettings;

    public WrappedSimulation(final String identifier, final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        super(identifier, settings, jobs, null, null, null);
        this.simSettings = (SimulationSettings) settings;
        this.stateExtractor = new VmManagementStateExtractor(this);
        this.actionDecoder = new VmManagementActionDecoder(this);
        this.rewardCalculator = new VmManagementRewardCalculator(this);
    }

    // ============== Abstract method implementations ==============

    @Override
    protected ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets) {
        return new CloudSimProxy((SimulationSettings) settings, cloudlets);
    }

    @Override
    protected int[] extractSecondaryObservation() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final int jobCoresWaiting = proxy.calculateJobCoresWaiting();
        final int largeVmPes = simSettings.getSmallVmPes() * simSettings.getLargeVmMultiplier();
        return new int[] { Math.min(jobCoresWaiting, largeVmPes) };
    }

    @Override
    protected Observation buildObservation(int[] infraObs, int[] secondaryObs) {
        return new Observation(infraObs, secondaryObs[0], new int[0]);
    }

    @Override
    protected SimulationStepInfo buildResetStepInfo() {
        return new SimulationStepInfo();
    }

    @Override
    protected SimulationStepInfo buildStepInfo(int[] actionResult,
            boolean terminated, boolean truncated) {
        final boolean isValid = actionResult[0] != -1;
        final double[] rewards = calculateReward(isValid);
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final int[] treeArray = simSettings.isSendObservationTreeArray()
                ? getInfrastructureObservation()
                : new int[0];
        return new SimulationStepInfo(rewards,
                proxy.getFinishedJobsWaitTimeLastTimestep(), getUnutilizedVmCoreRatio(),
                treeArray, actionResult[0], actionResult[1]);
    }

    @Override
    protected SimulationResetResult buildResetResult(Observation observation, SimulationStepInfo info) {
        return new SimulationResetResult(observation, info);
    }

    @Override
    protected SimulationStepResult buildStepResult(Observation observation, double reward,
            boolean terminated, boolean truncated, SimulationStepInfo info) {
        return new SimulationStepResult(observation, reward, terminated, truncated, info);
    }

    // ============== Domain-specific methods ==============

    private double getUnutilizedVmCoreRatio() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        List<Vm> vmList = proxy.getBroker().getVmExecList();
        Long unutilizedVmCores = getUnutilizedVmCores(vmList);
        Long runningVmCores = getRunningVmCores(vmList);
        return runningVmCores > 0 ? ((double) unutilizedVmCores / runningVmCores) : 0.0;
    }

    private Long getUnutilizedVmCores(List<Vm> vmList) {
        return vmList.parallelStream()
                .map(Vm::getExpectedFreePesNumber)
                .reduce(0L, Long::sum);
    }

    private Long getRunningVmCores(List<Vm> vmList) {
        return vmList.parallelStream()
                .map(Vm::getPesNumber)
                .reduce(0L, Long::sum);
    }

    int[] executeCustomAction(final int[] action) {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        if (action == null || action.length < 4) {
            String msg = "Invalid action: " +
                    (action == null ? "null" : "length=" + action.length);
            LOGGER.warn(msg);
            return new int[] {-1, 0};
        }

        if (action[0] == 1) {
            final int hostId = action[1];
            final int vmTypeIndex = action[3];
            final int vmCores = proxy.getVmCoreCountByType(simSettings.VM_TYPES[vmTypeIndex]);
            boolean isValid = addNewVm(simSettings.VM_TYPES[vmTypeIndex], hostId);
            if (!isValid) {
                return new int[] {-1, 0};
            }
            return new int[] {hostId, vmCores};
        }

        if (action[0] == 2) {
            final int vmIndex = action[2];
            List<Vm> vmList = proxy.getBroker().getVmExecList();
            if (vmIndex < 0 || vmIndex >= vmList.size()) {
                LOGGER.warn("destroy VM action invalid: vmIndex={} but only {} VMs running",
                        vmIndex, vmList.size());
                return new int[] {-1, 0};
            }
            Vm vm = vmList.get(vmIndex);
            int hostId = (int) vm.getHost().getId();
            int vmCores = (int) vm.getPesNumber();
            boolean isValid = removeVm(vmIndex);
            if (!isValid) {
                return new int[] {-1, 0};
            }
            return new int[] {hostId, vmCores};
        }

        return new int[] {0, 0};
    }

    private boolean removeVm(final int index) {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        if (!proxy.removeVm(index)) {
            LOGGER.debug("Removing a VM with index {} action is invalid. Ignoring.", index);
            return false;
        }
        return true;
    }

    private boolean addNewVm(final String type, final long hostId) {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        if (!proxy.addNewVm(type, hostId)) {
            LOGGER.warn("Adding a VM of type {} to host {} is invalid. Ignoring", type, hostId);
            return false;
        }
        return true;
    }

    int[] getInfrastructureObservation() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final int hostsNum = simSettings.getHostsCount();
        final int vmsNum = proxy.getBroker().getVmExecList().size();
        final int jobsNum = getRunningCloudletsCount().intValue();
        final int[] treeArray = new int[2 * (1 + hostsNum + vmsNum + jobsNum)];

        final int totalDatacenterCores = (int) simSettings.getDatacenterCores();
        final List<Host> hostList = proxy.getDatacenter().getHostList();
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
                    treeArray[currentIndex++] = 0;
                }
            }
        }
        return treeArray;
    }

    double[] calculateReward(final boolean isValid) {
        double[] rewards = new double[5];

        final double jobWaitCoef = simSettings.getRewardJobWaitCoef();
        final double runningVmCoresCoef = simSettings.getRewardRunningVmCoresCoef();
        final double unutilizedVmCoresCoef = simSettings.getRewardUnutilizedVmCoresCoef();
        final double invalidCoef = simSettings.getRewardInvalidCoef();

        final double jobWaitReward = -jobWaitCoef * getWaitingJobsRatio();
        final double runningVmCoresReward = -runningVmCoresCoef * getHostCoresAllocatedToVmsRatio();
        final double unutilizedVmCoresReward = -unutilizedVmCoresCoef * getUnutilizedVmCoreRatio();
        final double invalidReward = -invalidCoef * (isValid ? 0 : 1);

        double totalReward = 0;
        if (simSettings.getVmAllocationPolicy().equals("rule-based")) {
            totalReward = jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward;
        } else if (simSettings.getVmAllocationPolicy().equals("rl")) {
            totalReward = jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward + invalidReward;
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

    private double getWaitingJobsRatio() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final long arrivedJobsCount = proxy.getArrivedJobsCount();
        return arrivedJobsCount > 0
                ? proxy.getNotYetRunningJobsCount() / (double) arrivedJobsCount
                : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        return ((double) proxy.getAllocatedCores()) / simSettings.getTotalHostCores();
    }

    private Long getRunningCloudletsCount() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        List<Vm> vmList = proxy.getBroker().getVmExecList();
        return vmList.parallelStream()
                .map(Vm::getCloudletScheduler)
                .map(CloudletScheduler::getCloudletExecList)
                .mapToLong(List::size)
                .sum();
    }
}

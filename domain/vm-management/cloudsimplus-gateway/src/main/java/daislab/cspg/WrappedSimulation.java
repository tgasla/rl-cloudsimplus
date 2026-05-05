package daislab.cspg;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.List;

/**
 * VM Management simulation using RL-driven VM lifecycle management. Extends WrappedSimulationBase
 * with tree-array infrastructure observation and VM create/destroy action space.
 */
public class WrappedSimulation extends WrappedSimulationBase {

    // Concrete settings reference for domain-specific access
    private final SimulationSettings simSettings;

    public WrappedSimulation(final String identifier, final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        super(identifier, settings, jobs);
        this.simSettings = (SimulationSettings) settings;
    }

    // ============== Abstract method implementations ==============

    @Override
    protected ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets) {
        return new CloudSimProxy((SimulationSettings) settings, cloudlets);
    }

    @Override
    protected int[] extractInfrastructureObservation() {
        final CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
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

    @Override
    protected int[] extractSecondaryObservation() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final int jobCoresWaiting = proxy.calculateJobCoresWaiting();
        return new int[] {Math.min(jobCoresWaiting, simSettings.getMaxVmPes())};
    }

    @Override
    public SimulationStepResult step(final int[] action) {
        validateSimulationReset();
        currentStep++;
        LOGGER.info("Step {} starting", currentStep);

        final int[] actionResult = executeAction(action);
        final boolean isValid = actionResult[0] != -1;

        cloudSimProxy.runOneTimestep();

        final boolean terminated = !cloudSimProxy.isRunning();
        final boolean truncated = !terminated && (currentStep >= simSettings.getMaxEpisodeLength());

        // Single source of truth for reward — reuses isValid so the agent actually
        // sees the invalid-action penalty (previously the strategy path always passed true).
        final double[] rewards = calculateReward(isValid);
        final double reward = rewards[0];

        LOGGER.info("Step {} finished", currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", cloudSimProxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    cloudSimProxy.getFinishedJobsCount(), initialJobsDescriptors.size());
        }

        // Cache the obs so treeArray re-uses the same array without a second computation.
        final int[] infraObs = extractInfrastructureObservation();
        final Observation observation = buildObservation(infraObs, extractSecondaryObservation());

        final CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        final int[] treeArray = simSettings.isSendObservationTreeArray() ? infraObs : new int[0];
        final SimulationStepInfo info = new SimulationStepInfo(rewards,
                proxy.getFinishedJobsWaitTimeLastTimestep(), getUnutilizedVmCoreRatio(), treeArray,
                actionResult[0], actionResult[1]);

        return new SimulationStepResult(observation, reward, terminated, truncated, info);
    }

    // ============== Domain-specific methods ==============

    /**
     * Top-level action dispatch. RL/fromfile modes interpret the action array;
     * rule-based modes ignore it and let the policy decide.
     */
    int[] executeAction(final int[] action) {
        final String policy = simSettings.getVmAllocationPolicy();
        if (policy.equals("rl") || policy.equals("fromfile")) {
            return executeCustomAction(action);
        }
        executeRuleBasedAction();
        return new int[] {0, 0};
    }

    private void executeRuleBasedAction() {
        switch (simSettings.getVmAllocationPolicy()) {
            case "minimize-queue" -> executeMinimizeQueueAction();
            case "minimize-allocated" -> executeMinimizeAllocatedAction();
            case "minimize-unutilized" -> executeMinimizeUnutilizedAction();
            default -> throw new IllegalArgumentException(
                    "Not a rule-based vm_allocation_policy: "
                            + simSettings.getVmAllocationPolicy());
        }
    }

    private void executeMinimizeQueueAction() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        long maxCoresNeeded = proxy.calculateMaxJobCoresNeeded();
        long maxFreeCoresOnSameVm = proxy.getMaxFreeVmCores();
        boolean vmAvailable = maxFreeCoresOnSameVm >= maxCoresNeeded;
        if (!vmAvailable && maxCoresNeeded > 0) {
            proxy.getBroker().submitVmList(
                    proxy.createSingleVm(proxy.calculateTargetTime(), maxCoresNeeded));
        } else {
            proxy.destroyLargestIdleVm();
        }
    }

    private void executeMinimizeAllocatedAction() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        int coresNeeded = proxy.calculateJobCoresWaiting();
        int numTypes = simSettings.getVmTypesCount();
        // Descend from largest type: if coresNeeded < this type and it's running -> downsize;
        // if coresNeeded >= this type and it's not running -> create it.
        for (int i = numTypes - 1; i >= 0; i--) {
            long typeCores = simSettings.getVmCoreCountByTypeIndex(i);
            if (coresNeeded >= typeCores) {
                if (!proxy.isVmWithCoresRunning(typeCores)) {
                    proxy.getBroker().submitVmList(
                            proxy.createSingleVm(proxy.calculateTargetTime(), typeCores));
                } else {
                    proxy.destroyLargestIdleVm();
                }
                return;
            } else if (proxy.isVmWithCoresRunning(typeCores)) {
                proxy.destroyLargestIdleVm();
                return;
            }
        }
        proxy.destroyLargestIdleVm();
    }

    private void executeMinimizeUnutilizedAction() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        int numTypes = simSettings.getVmTypesCount();
        for (int i = 0; i < numTypes; i++) {
            long typeCores = simSettings.getVmCoreCountByTypeIndex(i);
            if (proxy.isJobWithCoresWaiting(typeCores) && !proxy.isVmWithCoresRunning(typeCores)) {
                proxy.getBroker().submitVmList(
                        proxy.createSingleVm(proxy.calculateTargetTime(), typeCores));
                return;
            }
        }
        proxy.destroyLargestIdleVm();
    }

    int[] executeCustomAction(final int[] action) {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        if (action == null || action.length < 4) {
            String msg = "Invalid action: " + (action == null ? "null" : "length=" + action.length);
            LOGGER.warn(msg);
            return new int[] {-1, 0};
        }

        if (action[0] == 1) {
            final int hostId = action[1];
            final int vmTypeIndex = action[3];
            final int vmCores = simSettings.getVmCoreCountByTypeIndex(vmTypeIndex);
            boolean isValid = addNewVm(vmTypeIndex, hostId);
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

    private boolean addNewVm(final int typeIndex, final long hostId) {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        if (!proxy.addNewVm(typeIndex, hostId)) {
            LOGGER.warn("Adding type-{} VM to host {} is invalid. Ignoring", typeIndex, hostId);
            return false;
        }
        return true;
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

        // reward_invalid_coef controls whether invalid actions are penalised:
        //   0.0 → masked RL (MaskablePPO guarantees all actions are valid; penalty never fires)
        //   >0  → ablation without masking (plain PPO; penalty fires on bad host/VM choices)
        // Rule-based and fromfile modes always produce valid actions, so invalidReward is 0
        // regardless of the coefficient.
        final double totalReward =
                jobWaitReward + runningVmCoresReward + unutilizedVmCoresReward + invalidReward;

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
        return arrivedJobsCount > 0 ? proxy.getNotYetRunningJobsCount() / (double) arrivedJobsCount
                : 0.0;
    }

    private double getHostCoresAllocatedToVmsRatio() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        return ((double) proxy.getAllocatedCores()) / simSettings.getDatacenterCores();
    }

    private Long getRunningCloudletsCount() {
        CloudSimProxy proxy = (CloudSimProxy) cloudSimProxy;
        List<Vm> vmList = proxy.getBroker().getVmExecList();
        return vmList.stream().map(Vm::getCloudletScheduler)
                .map(CloudletScheduler::getCloudletExecList).mapToLong(List::size).sum();
    }
}

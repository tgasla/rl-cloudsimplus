package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyBestFit;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CloudSimProxy extends CloudSimProxyBase {

    private SimulationSettings simSettings;
    private Datacenter datacenter;
    private VmCost vmCost;

    public CloudSimProxy(final SimulationSettings settings, final List<Cloudlet> inputJobs) {
        super(settings, inputJobs);
    }

    // ============== Abstract method implementations ==============

    @Override
    protected void setupInfrastructure() {
        simSettings = (SimulationSettings) settings;
        vmCost = new VmCost(simSettings);
        datacenter = createDatacenter();
        submitInitialVmList();
    }

    @Override
    protected Datacenter getPrimaryDatacenter() {
        return datacenter;
    }

    @Override
    protected void tryToSubmitJobs(final List<Cloudlet> cloudletList) {
        final List<Cloudlet> jobsToSubmit = new ArrayList<>();
        final double now = clock();
        final double targetTime = calculateTargetTime();

        LOGGER.info("[{} - {}]: Will try to submit {} jobs", now, targetTime, cloudletList.size());
        LOGGER.info("[{} - {}]: VMs running: {}", now, targetTime, broker.getVmExecList().size());
        for (Cloudlet cloudlet : cloudletList) {
            if (!isAnyVmSuitableForCloudlet(cloudlet)) {
                LOGGER.debug("[{} - {}]: Could not submit job {}, no suitable vm found",
                        now, targetTime, cloudlet.getId());
                continue;
            }
            cloudlet.setSubmissionDelay(Math.max(cloudlet.getSubmissionDelay() - now, 0));
            jobsToSubmit.add(cloudlet);
        }

        if (!jobsToSubmit.isEmpty()) {
            jobQueue.removeAll(jobsToSubmit);
            LOGGER.info("[{} - {}]: Submitting {} jobs", now, targetTime, jobsToSubmit.size());
            submitCloudletList(jobsToSubmit);
        }
    }

    // ============== Infrastructure creation ==============

    private void submitInitialVmList() {
        List<Vm> initialVmList = new ArrayList<>();
        for (int i = 0; i < simSettings.VM_TYPES.length; i++) {
            String vmType = simSettings.VM_TYPES[i];
            List<Vm> vmList = createVmList(simSettings.getInitialVmCounts()[i], vmType);
            initialVmList.forEach(v -> v.setDescription(vmType));
            initialVmList.addAll(vmList);
        }
        initialVmList.forEach(v -> vmCost.addNewVmToList(v));
        broker.submitVmList(initialVmList);
    }

    private Datacenter createDatacenter() {
        List<Host> hostList = createHostList();
        LOGGER.debug("Creating datacenter");
        return new DatacenterSimple(cloudSimPlus, hostList, defineVmAllocationPolicy());
    }

    private VmAllocationPolicy defineVmAllocationPolicy() {
        return switch (simSettings.getVmAllocationPolicy()) {
            case "rl", "fromfile" -> new VmAllocationPolicyCustom();
            case "rule-based" -> defineRuleBasedVmAllocationPolicy();
            default -> throw new IllegalArgumentException(
                    "Unknown VM allocation policy: " + simSettings.getVmAllocationPolicy());
        };
    }

    private VmAllocationPolicy defineRuleBasedVmAllocationPolicy() {
        return switch (simSettings.getAlgorithm()) {
            case "minimize-queue", "minimize-allocated", "minimize-unutilized" ->
                    new VmAllocationPolicyBestFit();
            default -> throw new IllegalArgumentException(
                    "Unknown algorithm: " + simSettings.getAlgorithm());
        };
    }

    private List<Host> createHostList() {
        List<Host> hostList = new ArrayList<>();
        final long hostRam = simSettings.getHostRam();
        final long hostBw = simSettings.getHostBw();
        final long hostStorage = simSettings.getHostStorage();
        for (int i = 0; i < simSettings.getHostsCount(); i++) {
            Host host = new HostWithoutCreatedList(hostRam, hostBw, hostStorage, createPeList())
                    .setRamProvisioner(new ResourceProvisionerSimple())
                    .setBwProvisioner(new ResourceProvisionerSimple())
                    .setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(host);
        }
        return hostList;
    }

    private List<Pe> createPeList() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < simSettings.getHostPes(); i++) {
            peList.add(new PeSimple(simSettings.getHostPeMips(), new PeProvisionerSimple()));
        }
        return peList;
    }

    private List<Vm> createVmList(final int vmCount, final String type) {
        List<Vm> vmList = new ArrayList<>(vmCount);
        for (int i = 0; i < vmCount; i++) {
            vmList.add(createVm(type));
        }
        return vmList;
    }

    private Vm createVm(final String type) {
        int sizeMultiplier = simSettings.getSizeMultiplier(type);
        Vm vm = new VmSimple(vmsCreated++, simSettings.getHostPeMips(),
                simSettings.getSmallVmPes() * sizeMultiplier);
        vm.setRam(simSettings.getSmallVmRam() * sizeMultiplier)
                .setBw(simSettings.getSmallVmBw())
                .setSize(simSettings.getSmallVmStorage())
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setShutDownDelay(simSettings.getVmShutdownDelay());
        vm.setSubmissionDelay(simSettings.getVmStartupDelay());
        vmCost.addNewVmToList(vm);
        return vm;
    }

    // ============== Rule-based actions ==============

    public boolean executeRuleBasedAction() {
        if (simSettings.getAlgorithm().equals("minimize-queue")) {
            executeMinimizeQueueAction();
        } else if (simSettings.getAlgorithm().equals("minimize-allocated")) {
            executeMinimizeAllocatedAction();
        } else if (simSettings.getAlgorithm().equals("minimize-unutilized")) {
            executeMinimizeUnutilizedAction();
        }
        return true;
    }

    private void executeMinimizeQueueAction() {
        long maxCoresNeeded = calculateMaxJobCoresNeeded();
        long maxFreeCoresOnSameVm = getMaxFreeVmCores();
        boolean vmAvailable = maxFreeCoresOnSameVm >= maxCoresNeeded;
        if (!vmAvailable && maxCoresNeeded > 0) {
            List<Vm> vmList = createSingleVm(calculateTargetTime(), maxCoresNeeded);
            broker.submitVmList(vmList);
        } else {
            destroyLargestIdleVm();
        }
    }

    private void executeMinimizeAllocatedAction() {
        int coresNeeded = calculateJobCoresWaiting();
        final long smallVmCores = getVmCoreCountByType(simSettings.VM_TYPES[0]);
        final long mediumVmCores = getVmCoreCountByType(simSettings.VM_TYPES[1]);
        final long largeVmCores =
                getVmCoreCountByType(simSettings.VM_TYPES[simSettings.VM_TYPES.length - 1]);
        if (coresNeeded >= largeVmCores && !isVmWithCoresRunning(largeVmCores)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), largeVmCores));
        } else if (coresNeeded >= mediumVmCores && isVmWithCoresRunning(largeVmCores)) {
            destroyLargestIdleVm();
        } else if (coresNeeded >= mediumVmCores && !isVmWithCoresRunning(mediumVmCores)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), mediumVmCores));
        } else if (coresNeeded >= smallVmCores && isVmWithCoresRunning(mediumVmCores)) {
            destroyLargestIdleVm();
        } else if (coresNeeded >= smallVmCores && !isVmWithCoresRunning(smallVmCores)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), smallVmCores));
        } else {
            destroyLargestIdleVm();
        }
    }

    private void executeMinimizeUnutilizedAction() {
        if (isJobWithCoresWaiting(2) && !isVmWithCoresRunning(2)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), 2));
        } else if (isJobWithCoresWaiting(4) && !isVmWithCoresRunning(4)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), 4));
        } else if (isJobWithCoresWaiting(8) && !isVmWithCoresRunning(8)) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), 8));
        } else {
            destroyLargestIdleVm();
        }
    }

    private List<Vm> createSingleVm(final double targetTime, final long coresNeeded) {
        final int vmTypesCount = simSettings.VM_TYPES.length;
        final List<Vm> vmList = new ArrayList<>();
        final double startTime = targetTime - simSettings.getTimestepInterval();

        int vmTypeIndex = vmTypesCount - 1;
        for (int i = 0; i < vmTypesCount; i++) {
            if (coresNeeded <= getVmCoreCountByType(simSettings.VM_TYPES[i])) {
                vmTypeIndex = i;
                break;
            }
        }

        if (vmTypeIndex == -1) {
            LOGGER.error("[{} - {}]: No VM type can fulfill {} cores needed",
                    startTime, targetTime, coresNeeded);
            return vmList;
        }

        final String vmType = simSettings.VM_TYPES[vmTypeIndex];
        LOGGER.info("[{} - {}]: {} VM cores needed, creating 1 {} VM",
                startTime, targetTime, coresNeeded, vmType);
        vmList.add(createVm(vmType).setDescription(vmType));
        return vmList;
    }

    private void destroyLargestIdleVm() {
        List<Vm> idleVms = broker.getVmExecList().stream()
                .filter(vm -> vm.getCloudletScheduler().isEmpty())
                .collect(Collectors.toList());
        idleVms.stream().max(Comparator.comparingLong(Vm::getPesNumber)).ifPresent(largestVm -> {
            cloudSimPlus.send(datacenter, datacenter, 0, CloudSimTag.VM_DESTROY, largestVm);
            LOGGER.info("No jobs to submit, destroying the largest idle VM");
        });
        if (idleVms.isEmpty()) {
            LOGGER.info("No idle VMs available for destruction.");
        }
    }

    // ============== VM add/remove ==============

    public boolean addNewVm(final String type, final long hostId) {
        LOGGER.debug("Agent action: Create a {} VM on host {}", type, hostId);
        final Host host = datacenter.getHostById(hostId);
        if (host == Host.NULL) {
            LOGGER.debug("VM creating ignored, no host with given id found");
            return false;
        }
        final Vm newVm = createVm(type);
        newVm.setDescription(type + "-" + hostId);
        if (!host.isSuitableForVm(newVm)) {
            LOGGER.debug("VM creating ignored, host not suitable");
            return false;
        }
        broker.submitVm(newVm);
        LOGGER.debug("Requested VM of type: {} at host {}", type, hostId);
        return true;
    }

    public boolean removeVm(final int index) {
        List<Vm> vmExecList = broker.getVmExecList();
        if (index >= vmExecList.size()) {
            LOGGER.warn("Can't kill VM with index {}: no such index found", index);
            return false;
        }
        destroyVm(vmExecList.get(index));
        return true;
    }

    private void destroyVm(Vm vm) {
        final String vmSize = vm.getDescription();
        final List<Cloudlet> execCloudlets =
                resetCloudlets(vm.getCloudletScheduler().getCloudletExecList());
        final List<Cloudlet> waitingCloudlets =
                resetCloudlets(vm.getCloudletScheduler().getCloudletWaitingList());
        final List<Cloudlet> affectedCloudlets =
                Stream.concat(execCloudlets.stream(), waitingCloudlets.stream())
                        .collect(Collectors.toList());
        datacenter.getVmAllocationPolicy().deallocateHostForVm(vm);
        vmCost.removeVmFromList(vm);
        LOGGER.info("{} Killing VM {} ({}), cloudlets to reschedule: {}",
                clock(), vm.getId(), vmSize, affectedCloudlets.size());
        if (!affectedCloudlets.isEmpty()) {
            rescheduleCloudlets(affectedCloudlets);
        }
    }

    private void rescheduleCloudlets(final List<Cloudlet> affectedCloudlets) {
        affectedCloudlets.forEach(cloudlet -> cloudlet.setSubmissionDelay(0));
        jobQueue.addAll(affectedCloudlets);
    }

    private Cloudlet resetCloudlet(final Cloudlet cloudlet) {
        return cloudlet.setVm(Vm.NULL).reset();
    }

    private List<Cloudlet> resetCloudlets(List<CloudletExecution> cloudlets) {
        return cloudlets.parallelStream()
                .map(CloudletExecution::getCloudlet)
                .map(this::resetCloudlet)
                .collect(Collectors.toList());
    }

    // ============== Domain-specific accessors ==============

    public int getVmCoreCountByType(final String type) {
        return simSettings.getSmallVmPes() * simSettings.getSizeMultiplier(type);
    }

    int calculateJobCoresWaiting() {
        return coresRequiredForJobs(getJobsToSubmitAtThisTimestep(calculateTargetTime()));
    }

    long calculateMaxJobCoresNeeded() {
        return getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .mapToLong(Cloudlet::getPesNumber).max().orElse(0);
    }

    private int coresRequiredForJobs(List<Cloudlet> jobs) {
        return (int) jobs.stream().mapToLong(Cloudlet::getPesNumber).sum();
    }

    private boolean isJobWithCoresWaiting(final long cores) {
        return getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .anyMatch(job -> job.getPesNumber() == cores);
    }

    private boolean isVmWithCoresRunning(final long cores) {
        return broker.getVmExecList().stream().anyMatch(vm -> vm.getPesNumber() == cores);
    }

    private long getMaxFreeVmCores() {
        return broker.getVmExecList().stream()
                .mapToLong(Vm::getExpectedFreePesNumber).max().orElse(0);
    }

    public Datacenter getDatacenter() {
        return datacenter;
    }

    public double getRunningCost() {
        return vmCost.getVMCostPerIteration(clock());
    }
}

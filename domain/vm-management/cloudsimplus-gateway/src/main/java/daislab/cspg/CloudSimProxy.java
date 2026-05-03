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
import java.util.Map;
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
    protected void printStats() {
        printCloudletStatus();
    }

    @Override
    protected void setupInfrastructure() {
        simSettings = (SimulationSettings) settings;
        vmCost = new VmCost(simSettings);
        datacenter = createDatacenter();
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
                LOGGER.debug("[{} - {}]: Could not submit job {}, no suitable vm found", now,
                        targetTime, cloudlet.getId());
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

    private Datacenter createDatacenter() {
        LOGGER.debug("Creating datacenter");
        return new DatacenterSimple(cloudSimPlus, createHostList(), defineVmAllocationPolicy());
    }

    private VmAllocationPolicy defineVmAllocationPolicy() {
        return switch (simSettings.getVmAllocationPolicy()) {
            case "rl", "fromfile" -> new VmAllocationPolicyCustom();
            case "bestfit" -> new VmAllocationPolicyBestFit();
            default -> throw new IllegalArgumentException(
                    "Unknown VM allocation policy: " + simSettings.getVmAllocationPolicy());
        };
    }

    @SuppressWarnings("unchecked")
    private List<Host> createHostList() {
        List<Host> hostList = new ArrayList<>();
        for (Map<String, Object> dcMap : simSettings.getDatacenters()) {
            int dcAmount = ((Number) dcMap.getOrDefault("amount", 1)).intValue();
            List<Map<String, Object>> hostMaps = (List<Map<String, Object>>) dcMap.get("hosts");
            for (Map<String, Object> hostMap : hostMaps) {
                int amount = dcAmount * ((Number) hostMap.get("amount")).intValue();
                long ram = ((Number) hostMap.get("ram")).longValue();
                long bw = ((Number) hostMap.get("bw")).longValue();
                long storage = ((Number) hostMap.get("storage")).longValue();
                int pes = ((Number) hostMap.get("pes")).intValue();
                long peMips = ((Number) hostMap.get("pe_mips")).longValue();
                for (int i = 0; i < amount; i++) {
                    hostList.add(
                            new HostWithoutCreatedList(ram, bw, storage, createPeList(pes, peMips))
                                    .setRamProvisioner(new ResourceProvisionerSimple())
                                    .setBwProvisioner(new ResourceProvisionerSimple())
                                    .setVmScheduler(new VmSchedulerTimeShared()));
                }
            }
        }
        return hostList;
    }

    private List<Pe> createPeList(final int pes, final long mips) {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < pes; i++) {
            peList.add(new PeSimple(mips, new PeProvisionerSimple()));
        }
        return peList;
    }

    private Vm createVm(final int typeIndex) {
        Map<String, Object> typeConfig = simSettings.getVmTypeConfig(typeIndex);
        long pes = ((Number) typeConfig.get("pes")).longValue();
        long peMips = ((Number) typeConfig.get("pe_mips")).longValue();
        long ram = ((Number) typeConfig.get("ram")).longValue();
        long size = ((Number) typeConfig.get("size")).longValue();
        long bw = ((Number) typeConfig.get("bw")).longValue();
        Vm vm = new VmSimple(vmsCreated++, peMips, pes);
        vm.setRam(ram).setSize(size).setBw(bw)
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setShutDownDelay(simSettings.getVmShutdownDelay());
        vm.setSubmissionDelay(simSettings.getVmStartupDelay());
        vmCost.addNewVmToList(vm);
        return vm;
    }

    // ============== Rule Based actions ==============

    public boolean executeRuleBasedAction() {
        switch (simSettings.getAlgorithm()) {
            case "minimize-queue" -> executeMinimizeQueueAction();
            case "minimize-allocated" -> executeMinimizeAllocatedAction();
            case "minimize-unutilized" -> executeMinimizeUnutilizedAction();
            default -> throw new IllegalArgumentException(
                    "Unknown algorithm: " + simSettings.getAlgorithm());
        }
        return true;
    }


    private void executeMinimizeQueueAction() {
        long maxCoresNeeded = calculateMaxJobCoresNeeded();
        long maxFreeCoresOnSameVm = getMaxFreeVmCores();
        boolean vmAvailable = maxFreeCoresOnSameVm >= maxCoresNeeded;
        if (!vmAvailable && maxCoresNeeded > 0) {
            broker.submitVmList(createSingleVm(calculateTargetTime(), maxCoresNeeded));
        } else {
            destroyLargestIdleVm();
        }
    }

    private void executeMinimizeAllocatedAction() {
        int coresNeeded = calculateJobCoresWaiting();
        int numTypes = simSettings.getVmTypesCount();
        // Descend from largest type: if coresNeeded < this type and it's running → downsize;
        // if coresNeeded >= this type and it's not running → create it.
        for (int i = numTypes - 1; i >= 0; i--) {
            long typeCores = simSettings.getVmCoreCountByTypeIndex(i);
            if (coresNeeded >= typeCores) {
                if (!isVmWithCoresRunning(typeCores)) {
                    broker.submitVmList(createSingleVm(calculateTargetTime(), typeCores));
                } else {
                    destroyLargestIdleVm();
                }
                return;
            } else if (isVmWithCoresRunning(typeCores)) {
                destroyLargestIdleVm();
                return;
            }
        }
        destroyLargestIdleVm();
    }

    private void executeMinimizeUnutilizedAction() {
        int numTypes = simSettings.getVmTypesCount();
        for (int i = 0; i < numTypes; i++) {
            long typeCores = simSettings.getVmCoreCountByTypeIndex(i);
            if (isJobWithCoresWaiting(typeCores) && !isVmWithCoresRunning(typeCores)) {
                broker.submitVmList(createSingleVm(calculateTargetTime(), typeCores));
                return;
            }
        }
        destroyLargestIdleVm();
    }

    private List<Vm> createSingleVm(final double targetTime, final long coresNeeded) {
        final int numTypes = simSettings.getVmTypesCount();
        final List<Vm> vmList = new ArrayList<>();
        final double startTime = targetTime - simSettings.getTimestepInterval();

        int typeIndex = numTypes - 1;
        for (int i = 0; i < numTypes; i++) {
            if (coresNeeded <= simSettings.getVmCoreCountByTypeIndex(i)) {
                typeIndex = i;
                break;
            }
        }

        LOGGER.info("[{} - {}]: {} VM cores needed, creating 1 type-{} VM", startTime, targetTime,
                coresNeeded, typeIndex);
        vmList.add(createVm(typeIndex));
        return vmList;
    }

    private void destroyLargestIdleVm() {
        List<Vm> idleVms = broker.getVmExecList().stream()
                .filter(vm -> vm.getCloudletScheduler().isEmpty()).collect(Collectors.toList());
        idleVms.stream().max(Comparator.comparingLong(Vm::getPesNumber)).ifPresent(largestVm -> {
            cloudSimPlus.send(datacenter, datacenter, 0, CloudSimTag.VM_DESTROY, largestVm);
            LOGGER.info("No jobs to submit, destroying the largest idle VM");
        });
        if (idleVms.isEmpty()) {
            LOGGER.info("No idle VMs available for destruction.");
        }
    }

    // ============== VM add/remove ==============

    public boolean addNewVm(final int typeIndex, final long hostId) {
        LOGGER.debug("Agent action: Create type-{} VM on host {}", typeIndex, hostId);
        final Host host = datacenter.getHostById(hostId);
        if (host == Host.NULL) {
            LOGGER.debug("VM creating ignored, no host with given id found");
            return false;
        }
        final Vm newVm = createVm(typeIndex);
        if (!host.isSuitableForVm(newVm)) {
            LOGGER.debug("VM creating ignored, host not suitable");
            return false;
        }
        broker.submitVm(newVm);
        LOGGER.debug("Requested type-{} VM at host {}", typeIndex, hostId);
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
        final List<Cloudlet> execCloudlets =
                resetCloudlets(vm.getCloudletScheduler().getCloudletExecList());
        final List<Cloudlet> waitingCloudlets =
                resetCloudlets(vm.getCloudletScheduler().getCloudletWaitingList());
        final List<Cloudlet> affectedCloudlets =
                Stream.concat(execCloudlets.stream(), waitingCloudlets.stream())
                        .collect(Collectors.toList());
        datacenter.getVmAllocationPolicy().deallocateHostForVm(vm);
        vmCost.removeVmFromList(vm);
        LOGGER.info("{} Killing VM {} ({} PEs), cloudlets to reschedule: {}", clock(), vm.getId(),
                vm.getPesNumber(), affectedCloudlets.size());
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
        return cloudlets.parallelStream().map(CloudletExecution::getCloudlet)
                .map(this::resetCloudlet).collect(Collectors.toList());
    }

    // ============== Domain-specific accessors ==============

    int calculateJobCoresWaiting() {
        return (int) getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .mapToLong(Cloudlet::getPesNumber).sum();
    }

    private boolean isJobWithCoresWaiting(final long cores) {
        return getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .anyMatch(job -> job.getPesNumber() == cores);
    }

    private boolean isVmWithCoresRunning(final long cores) {
        return broker.getVmExecList().stream().anyMatch(vm -> vm.getPesNumber() == cores);
    }

    private long getMaxFreeVmCores() {
        return broker.getVmExecList().stream().mapToLong(Vm::getExpectedFreePesNumber).max()
                .orElse(0);
    }

    public Datacenter getDatacenter() {
        return datacenter;
    }

    public double getRunningCost() {
        return vmCost.getVMCostPerIteration(clock());
    }
}

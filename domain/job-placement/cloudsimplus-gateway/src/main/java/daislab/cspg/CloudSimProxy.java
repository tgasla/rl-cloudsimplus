package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyBestFit;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CloudSimProxy extends CloudSimProxyBase {

    private SimulationSettings simSettings;
    private List<Datacenter> datacenters;

    public CloudSimProxy(final SimulationSettings settings, final List<Cloudlet> inputJobs) {
        super(settings, inputJobs);
    }

    // ============== Abstract method implementations ==============

    @Override
    protected void setupInfrastructure() {
        simSettings = (SimulationSettings) settings;
        datacenters = createDatacenters(simSettings.getDatacenters());
    }

    @Override
    protected Datacenter getPrimaryDatacenter() {
        return datacenters.get(0);
    }

    @Override
    protected void tryToSubmitJobs(final List<Cloudlet> cloudletList) {
        final List<Cloudlet> jobsToSubmit = new ArrayList<>();
        final double targetTime = calculateTargetTime();

        LOGGER.info("[{} - {}): Will try to submit {} jobs", clock(), targetTime,
                cloudletList.size());
        LOGGER.info("[{} - {}): VMs running: {}", clock(), targetTime,
                broker.getVmExecList().size());
        for (Cloudlet cloudlet : cloudletList) {
            if (cloudlet.getVm() == null || cloudlet.getVm() == Vm.NULL) {
                LOGGER.info("[{} - {}): Cloudlet {} not submitted because action was no-op.",
                        clock(), targetTime, cloudlet.getId());
                continue;
            }
            cloudlet.setSubmissionDelay(Math.max(cloudlet.getSubmissionDelay() - clock(), 0));
            LOGGER.info("[{} - {}): Submitting cloudlet {} with delay {}", clock(), targetTime,
                    cloudlet.getId(), cloudlet.getSubmissionDelay());
            jobsToSubmit.add(cloudlet);
        }

        if (!jobsToSubmit.isEmpty()) {
            jobQueue.removeAll(jobsToSubmit);
            LOGGER.info("[{} - {}): Submitting {} jobs", clock(), targetTime, jobsToSubmit.size());
            submitCloudletList(jobsToSubmit);
        }
    }

    @Override
    protected void printStats() {
        printCloudletProgress();
    }

    // ============== Infrastructure creation ==============

    private List<Datacenter> createDatacenters(final List<Map<String, Object>> listOfDcMaps) {
        List<Datacenter> dcs = new ArrayList<>();
        for (Map<String, Object> dcMap : listOfDcMaps) {
            final int dcAmount = parseInt(dcMap.get("amount"));
            for (int i = 0; i < dcAmount; i++) {
                dcs.add(createDatacenter(dcMap));
            }
        }
        return dcs;
    }

    @SuppressWarnings("unchecked")
    private Datacenter createDatacenter(final Map<String, Object> dcMap) {
        // job-placement always uses bestfit for VM-to-host placement; the RL agent operates
        // at the higher level of cloudlet-to-DC mapping, not VM-to-host.
        final VmAllocationPolicy vmAllocationPolicy = new VmAllocationPolicyBestFit();
        final List<Map<Host, List<Vm>>> hostVmMapping = createHostVmMapping(
                (List<Map<String, Object>>) dcMap.get("hosts"), vmAllocationPolicy);
        final String type = String.valueOf(dcMap.get("type"));
        final List<Integer> connectTo = (List<Integer>) dcMap.get("connect_to");
        LOGGER.info("while creating datacenter, I have connectTo {}", connectTo.toString());
        final List<Host> hostList = getHostListFromMapping(hostVmMapping);
        final Datacenter dc =
                new DatacenterWithType(cloudSimPlus, hostList, vmAllocationPolicy, type, connectTo);
        LOGGER.info("Datacenter created: {}", dc.getId());
        allocateHostsForVms(hostVmMapping, vmAllocationPolicy);
        broker.submitVmList(getVmsFromAllHosts(hostVmMapping));
        return dc;
    }

    private List<Map<Host, List<Vm>>> createHostVmMapping(
            final List<Map<String, Object>> listOfHostMaps,
            final VmAllocationPolicy vmAllocationPolicy) {
        List<Map<Host, List<Vm>>> hostVmMapping = new ArrayList<>();
        for (Map<String, Object> hostMap : listOfHostMaps) {
            final long hostRam = parseLong(hostMap.get("ram"));
            final long hostStorage = parseLong(hostMap.get("storage"));
            final long hostBw = parseLong(hostMap.get("bw"));
            final List<Pe> peList =
                    createPeList(parseLong(hostMap.get("pes")), parseLong(hostMap.get("pe_mips")));
            final int hostAmount = parseInt(hostMap.get("amount"));
            for (int i = 0; i < hostAmount; i++) {
                final Host host = new HostSimple(hostRam, hostBw, hostStorage, peList)
                        .setRamProvisioner(new ResourceProvisionerSimple())
                        .setBwProvisioner(new ResourceProvisionerSimple())
                        .setVmScheduler(new VmSchedulerTimeShared());
                @SuppressWarnings("unchecked")
                final List<Vm> vms =
                        createVmList((List<Map<String, Object>>) hostMap.get("vms"));
                hostVmMapping.add(Map.of(host, vms));
            }
        }
        return hostVmMapping;
    }

    private void allocateHostsForVms(final List<Map<Host, List<Vm>>> vmToHostMapList,
            final VmAllocationPolicy vmAllocationPolicy) {
        for (Map<Host, List<Vm>> vmToHostMap : vmToHostMapList) {
            for (Map.Entry<Host, List<Vm>> entry : vmToHostMap.entrySet()) {
                entry.getValue()
                        .forEach(vm -> vmAllocationPolicy.allocateHostForVm(vm, entry.getKey()));
            }
        }
    }

    private List<Host> getHostListFromMapping(final List<Map<Host, List<Vm>>> hostVmMapping) {
        return hostVmMapping.stream().map(Map::keySet).flatMap(Set::stream)
                .collect(Collectors.toList());
    }

    private List<Vm> getVmsFromAllHosts(final List<Map<Host, List<Vm>>> vmToHostMapList) {
        return vmToHostMapList.stream()
                .flatMap(entry -> entry.values().stream().flatMap(List::stream))
                .collect(Collectors.toList());
    }

    private List<Vm> createVmList(final List<Map<String, Object>> listOfVmMaps) {
        List<Vm> vmList = new ArrayList<>();
        for (Map<String, Object> vmMap : listOfVmMaps) {
            final int vmAmount = parseInt(vmMap.get("amount"));
            for (int i = 0; i < vmAmount; i++) {
                vmList.add(createVm(parseLong(vmMap.get("pes")), parseLong(vmMap.get("pe_mips")),
                        parseLong(vmMap.get("ram")), parseLong(vmMap.get("size")),
                        parseLong(vmMap.get("bw"))));
            }
        }
        return vmList;
    }

    private Vm createVm(final long pes, final long pe_mips, final long ram, final long size,
            final long bw) {
        Vm vm = new VmSimple(vmsCreated++, pe_mips, pes);
        vm.setRam(ram).setSize(size).setBw(bw)
                .setCloudletScheduler(new OptimizedCloudletScheduler());
        return vm;
    }

    // ============== Domain-specific observation helpers ==============

    static final int JOB_OBS_FEATURES = 4; // cores, location, delaySensitivity, deadline

    int[] getJobsWaitingObservation() {
        final double targetTime = calculateTargetTime();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        int[] jobsWaitingObs = new int[JOB_OBS_FEATURES * jobsToSubmitList.size()];
        for (int i = 0; i < jobsToSubmitList.size(); i++) {
            CloudletWithLocation job = (CloudletWithLocation) jobsToSubmitList.get(i);
            jobsWaitingObs[JOB_OBS_FEATURES * i] = (int) job.getPesNumber();
            jobsWaitingObs[JOB_OBS_FEATURES * i + 1] = job.getLocation();
            jobsWaitingObs[JOB_OBS_FEATURES * i + 2] = job.getDelaySensitivity();
            jobsWaitingObs[JOB_OBS_FEATURES * i + 3] = job.getDeadline();
        }
        return jobsWaitingObs;
    }

    int[] calculateCoresWaitingPerJob() {
        return getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .mapToInt(c -> (int) c.getPesNumber()).toArray();
    }

    int calculateTotalJobCoresWaiting() {
        return (int) getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .mapToLong(Cloudlet::getPesNumber).sum();
    }

    long getQueuedJobsCount() {
        return inputJobs.parallelStream().filter(c -> jobArrivalTimeMap.get(c.getId()) < clock())
                .filter(c -> c.getStatus().equals(Cloudlet.Status.QUEUED)).count();
    }

    // ============== Datacenter accessors ==============

    public Datacenter getDatacenterById(final int id) {
        return cloudSimPlus.getCis().getDatacenterList().stream().filter(dc -> dc.getId() == id)
                .findFirst().orElse(Datacenter.NULL);
    }

    public Datacenter getDatacenterByIdx(final int idx) {
        return datacenters.get(idx);
    }

    public List<Cloudlet> getSimulationCloudletList() {
        return inputJobs;
    }

    // ============== Debug / diagnostics ==============

    void printCloudletProgress() {
        for (Datacenter dc : cloudSimPlus.getCis().getDatacenterList()) {
            for (Host host : dc.getHostList()) {
                for (Vm vm : host.getVmList()) {
                    for (CloudletExecution ce : vm.getCloudletScheduler().getCloudletExecList()) {
                        LOGGER.info(
                                "Cloudlet {}, {} / {} executed. Total length: {}. Host {} in DC {}. HostPes: {}, HostMips: {}",
                                ce.getCloudlet().getId(), ce.getCloudlet().getFinishedLengthSoFar(),
                                ce.getCloudlet().getLength(), ce.getCloudlet().getTotalLength(),
                                host.getId(), host.getDatacenter().getId(), host.getPesNumber(),
                                host.getMips());
                    }
                }
            }
        }
    }

    // ============== Private helpers ==============

    private long parseLong(Object obj) {
        return ((Number) obj).longValue();
    }

    private int parseInt(Object obj) {
        return ((Number) obj).intValue();
    }
}

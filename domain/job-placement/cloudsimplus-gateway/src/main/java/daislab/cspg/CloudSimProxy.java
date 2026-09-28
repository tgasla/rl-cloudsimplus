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
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CloudSimProxy extends CloudSimProxyBase {

    private SimulationSettings simSettings;
    private List<Datacenter> datacenters;
    // Submitted jobs still crossing the network to their VM, with the time they reach it.
    // CloudSim leaves a cloudlet INSTANTIATED until its VM's scheduler receives it.
    private final Map<Cloudlet, Double> inFlight = new LinkedHashMap<>();

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
            // Wait for the job to arrive if it arrives later in this step, then cross the network
            // from the moment of binding. Relative to now, so deferring a job never makes the
            // delay disappear.
            final double delay = Math.max(cloudlet.getSubmissionDelay() - clock(), 0)
                    + simSettings.networkDelay(getDatacenterType(cloudlet.getVm()));
            cloudlet.setSubmissionDelay(delay);
            inFlight.put(cloudlet, clock() + delay);
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
        // JSON numbers deserialise to Double, and the unchecked cast to List<Integer> is a
        // no-op under erasure, so the list silently holds Doubles until something extracts
        // an element. Normalise here so every consumer really does get Integers.
        final List<?> rawConnectTo = (List<?>) dcMap.get("connect_to");
        final List<Integer> connectTo = rawConnectTo == null ? List.of()
                : rawConnectTo.stream().map(v -> ((Number) v).intValue()).toList();
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

    // gRPC wire format per job: [cores, location, nominal_runtime_ref, time_to_due, s0, s1, s2].
    // Python strips location (idx 1) before exposing the job to the policy and uses it for the
    // reachability mask. s0..s2 one-hot the delay sensitivity (tolerant, moderate, critical).
    // Must match Python JobPlacementEnv._JOB_WIRE_FEATURES.
    static final int JOB_OBS_FEATURES = 7;
    static final int SENSITIVITY_LEVELS = 3;

    /**
     * The jobs the agent decides on this timestep: the first max_jobs_waiting arrived, unsubmitted
     * jobs by (due time, arrival, id). Observation slot i and action[i] both refer to element i.
     * Anything that aggregates over the backlog must use getJobsToSubmitAtThisTimestep instead.
     */
    List<Cloudlet> getVisibleJobs(final double targetTime) {
        return getJobsToSubmitAtThisTimestep(targetTime).stream()
                .sorted(Comparator.comparingDouble(this::getDueTime)
                        .thenComparingDouble(c -> jobArrivalTimeMap.get(c.getId()))
                        .thenComparingLong(Cloudlet::getId))
                .limit(simSettings.getMaxJobsWaiting())
                .toList();
    }

    /** Completion deadline: arrival plus the job's deadline in timesteps. */
    double getDueTime(final Cloudlet job) {
        return jobArrivalTimeMap.get(job.getId())
                + ((CloudletWithLocation) job).getDeadline() * settings.getTimestepInterval();
    }

    int[] getJobsWaitingObservation() {
        final List<Cloudlet> visibleJobs = getVisibleJobs(calculateTargetTime());
        final double interval = settings.getTimestepInterval();
        final double refMiPerTimestep = simSettings.getMipsRef() * interval;
        final int[] jobsWaitingObs = new int[JOB_OBS_FEATURES * visibleJobs.size()];
        for (int i = 0; i < visibleJobs.size(); i++) {
            final CloudletWithLocation job = (CloudletWithLocation) visibleJobs.get(i);
            final int sensitivity = job.getDelaySensitivity();
            if (sensitivity < 0 || sensitivity >= SENSITIVITY_LEVELS) {
                throw new IllegalStateException("Job " + job.getId()
                        + " has delay sensitivity " + sensitivity + ", expected 0.."
                        + (SENSITIVITY_LEVELS - 1));
            }
            final int base = JOB_OBS_FEATURES * i;
            jobsWaitingObs[base] = (int) job.getPesNumber();
            jobsWaitingObs[base + 1] = job.getLocation();
            // Cloudlet length is per PE, so this is the runtime on a reference-speed PE.
            jobsWaitingObs[base + 2] = (int) Math.ceil(job.getLength() / refMiPerTimestep);
            jobsWaitingObs[base + 3] =
                    (int) Math.max(0, Math.floor((getDueTime(job) - clock()) / interval));
            jobsWaitingObs[base + 4 + sensitivity] = 1;
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

    /** Removes jobs from the queue without submitting them (expired, never placed). */
    void evict(final List<Cloudlet> jobs) {
        jobQueue.removeAll(jobs);
    }

    /** Forgets jobs that have reached their VM; call after the clock advances. */
    void pruneInFlight() {
        inFlight.keySet().removeIf(c -> c.getStatus() != Cloudlet.Status.INSTANTIATED);
    }

    /** Jobs submitted to vm that have not reached it yet, with their arrival time at it. */
    Map<Cloudlet, Double> getInFlight(final Vm vm) {
        final Map<Cloudlet, Double> toVm = new LinkedHashMap<>();
        inFlight.forEach((cloudlet, arrival) -> {
            if (cloudlet.getVm() == vm) {
                toVm.put(cloudlet, arrival);
            }
        });
        return toVm;
    }

    static String getDatacenterType(final Vm vm) {
        return ((DatacenterWithType) vm.getHost().getDatacenter()).getType();
    }

    // ============== Drift injection ==============

    /**
     * Simulates a datacenter failure by marking all its hosts as failed.
     * The observation will reflect 0 free PEs for the failed DC, and action masking
     * will prevent the agent from placing further jobs there.
     */
    public void injectDrift(final int dcIndex) {
        if (dcIndex < 0 || dcIndex >= datacenters.size()) {
            LOGGER.warn("injectDrift: invalid DC index {} (total DCs: {})", dcIndex, datacenters.size());
            return;
        }
        final Datacenter dc = datacenters.get(dcIndex);
        LOGGER.info("Drift: failing all hosts in DC {} (index {})", dc.getId(), dcIndex);
        dc.getHostList().forEach(host -> host.setFailed(true));
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

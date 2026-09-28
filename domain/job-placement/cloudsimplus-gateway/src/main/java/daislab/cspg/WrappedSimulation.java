package daislab.cspg;

import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.PriorityQueue;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.stream.Collectors;


public class WrappedSimulation extends WrappedSimulationBase {

    // Must match Python JobPlacementEnv.HOST_OBS_FEATURES:
    // [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts]
    static final int HOST_OBS_FEATURES = 5;

    // Concrete settings reference for domain-specific access
    private final SimulationSettings simSettings;

    // RL-level episode tracking (RL concepts; not present in CloudSimProxy)
    private double bestEpisodeReward;
    private double currentEpisodeReward;
    private double lastReward = 0.0;

    // Net-SLA accounting of the current episode, and the shaping potential of the last state
    private SlaLedger ledger;
    private double lastPotential;
    // Per DC index: the fastest PE, and the job origins (by DC index) each DC can serve
    private double[] dcPeMips;
    private List<List<Integer>> reachableDcsByLocation;

    // Action-phase placement counter: set by bind(), consumed in step info
    private int jobsPlacedThisTimestep;

    public WrappedSimulation(final String identifier, final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        super(identifier, settings, jobs);
        this.simSettings = (SimulationSettings) settings;
        bestEpisodeReward = Double.NEGATIVE_INFINITY;
    }

    // ============== Abstract method implementations ==============

    @Override
    protected ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets) {
        return new CloudSimProxy(simSettings, cloudlets);
    }

    @Override
    protected int[] extractInfrastructureObservation() {
        switch (simSettings.getStateSpaceType()) {
            case "dcid-dctype-vmcap-freepes-backlog-per-host":
                return getInfraObsPerHost();
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
    public SimulationResetResult reset(final long seed, final String jobsJson) {
        this.currentEpisodeReward = 0;
        final SimulationResetResult result = super.reset(seed, jobsJson);
        ledger = new SlaLedger(proxy().getSimulationCloudletList(), proxy().jobArrivalTimeMap,
                simSettings);
        cacheTopology();
        lastPotential = potential();
        return result;
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

        jobsPlacedThisTimestep = 0;
        executeCustomCloudletToDcAction(action);

        final double targetTime = proxy.calculateTargetTime();
        final int jobsWaiting = proxy.getJobsToSubmitAtThisTimestep(targetTime).size();

        proxy.runOneTimestep();
        ledger.beginStep();
        proxy.evict(ledger.resolve(clock()));
        if (currentStep >= simSettings.getMaxEpisodeLength()) {
            drainToResolution();
        }

        // Every job resolves, so the episode ends when none is left, at the latest at the horizon.
        final boolean terminated = !ledger.anyUnresolved();
        final double unshapedReward = ledger.stepReward();
        final double potential = terminated ? 0 : potential();
        double reward = unshapedReward;
        if (simSettings.isRewardShaping()) {
            reward += simSettings.getRewardShapingGamma() * potential - lastPotential;
        }
        lastPotential = potential;
        this.lastReward = reward;
        this.currentEpisodeReward += reward;

        LOGGER.info("Step {} finished, reward {}", currentStep, reward);
        LOGGER.debug("Length of future events queue: {}", proxy.getNumberOfFutureEvents());
        if (terminated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    proxy.getBroker().getCloudletFinishedList().size(),
                    proxy.getSimulationCloudletList().size());
            if (currentEpisodeReward > bestEpisodeReward) {
                bestEpisodeReward = currentEpisodeReward;
                LOGGER.info("New best episode reward: {}", bestEpisodeReward);
            }
        }

        SimulationStepInfo info = new SimulationStepInfo(jobsWaiting, jobsPlacedThisTimestep,
                calculateJobsPlacedRatio(jobsPlacedThisTimestep, jobsWaiting),
                proxy.getFinishedJobsWaitTimeLastTimestep(), ledger.getValueRealized(),
                ledger.getPenaltyPaid(), ledger.getResourceCost(), ledger.getJobsMet(),
                ledger.getJobsViolated(), ledger.getJobsExpiredUnplaced(), potential,
                ledger.getOfferedValue(), unshapedReward);

        final Observation observation =
                buildObservation(extractInfrastructureObservation(), extractSecondaryObservation());
        return new SimulationStepResult(observation, reward, terminated, false, info);
    }

    /**
     * At the horizon the agent stops deciding: every unplaced job is violated, and the simulation
     * runs on until each placed job has finished or passed its due time, so a placement is
     * scored by what actually happened to it rather than cut off mid-run.
     */
    private void drainToResolution() {
        proxy().evict(ledger.expireAllUnplaced());
        while (ledger.anyUnresolved()) {
            proxy().runOneTimestep();
            ledger.resolve(clock());
        }
    }

    /** Binds a job to a VM. Every action path goes through here, so all policies are scored alike. */
    private void bind(final Cloudlet job, final Vm vm) {
        proxy().getBroker().bindCloudletToVm(job, vm);
        final String dcType = CloudSimProxy.getDatacenterType(vm);
        // Priced per reference core-second: the same job costs the same wherever it is fast.
        final double cost = simSettings.costPerRefCoreSecond(dcType) * job.getPesNumber()
                * job.getLength() / simSettings.getMipsRef();
        ledger.onBind(job, cost);
        jobsPlacedThisTimestep++;
    }

    // Casts the inherited cloudSimProxy field to the concrete type used by jp
    private CloudSimProxy proxy() {
        return (CloudSimProxy) cloudSimProxy;
    }

    /**
     * The VM of the DC with the most expected free PEs, even if that is fewer than the job needs:
     * a full DC queues the job instead of refusing it, so congestion is a priced decision (a
     * later finish) rather than a hard constraint. Only VMs too small to ever hold it are skipped.
     */
    private Vm getMostFreeVmOfDcForCloudlet(final int targetDcId, final Cloudlet cloudlet) {
        long maxExpectedFreePes = Long.MIN_VALUE;
        Vm mostFreeVm = Vm.NULL;
        final double targetTime = proxy().calculateTargetTime();
        List<Vm> vmList = proxy().getBroker().getVmExecList();
        List<Cloudlet> cloudletList = proxy().getJobsToSubmitAtThisTimestep(targetTime);

        Map<Vm, Long> expectedToUseVmPesMap =
                vmList.stream().collect(Collectors.toMap(vm -> vm, vm -> cloudletList.stream()
                        .filter(c -> c.getVm() == vm).mapToLong(Cloudlet::getPesNumber).sum()));

        for (Vm vm : vmList) {
            final int dcId = (int) vm.getHost().getDatacenter().getId();
            if (dcId != targetDcId) {
                continue;
            }
            final long usedVmPes = vm.getCloudletScheduler().getCloudletList().stream()
                    .mapToLong(Cloudlet::getPesNumber).sum()
                    + proxy().getInFlight(vm).keySet().stream()
                            .mapToLong(Cloudlet::getPesNumber).sum();
            // Negative once jobs queue on the VM: it then measures how overloaded the VM is.
            final long expectedFreePes =
                    vm.getPesNumber() - usedVmPes - expectedToUseVmPesMap.get(vm);

            if (vm.isSuitableForCloudlet(cloudlet)) {
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

    private void executeCustomCloudletToDcAction(final int[] action) {
        switch (simSettings.getCloudletToDcMapping()) {
            case "rl" -> executeRlCloudletToDcAction(action);
            case "earliest-shortest-to-most-free-dc" -> executeEarliestShortestCloudletToMostFreeDcAction();
            case "earliest-most-critical-to-nearest-dc" -> executeEarliestMostCriticalCloudletToNearestDcAction();
            default -> throw new IllegalArgumentException("Unknown cloudlet_to_dc_mapping: "
                    + simSettings.getCloudletToDcMapping());
        }
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
            // Index by the connectTo entry, not the loop counter. These coincide only
            // while connectTo indices happen to be contiguous from 0; any topology whose
            // connect_to skips an index would otherwise select the wrong datacenters.
            final int connectedDcIndex = connectToArray.get(i);
            DatacenterWithType connectedDatacenter =
                    (DatacenterWithType) datacenterList.get(connectedDcIndex);
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

    private void executeEarliestMostCriticalCloudletToNearestDcAction() {
        final double targetTime = proxy().calculateTargetTime();
        final List<Cloudlet> jobsWaitingList = proxy().getJobsToSubmitAtThisTimestep(targetTime);
        final List<Cloudlet> jobsToProcessList = new ArrayList<>(jobsWaitingList);

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
                    bind(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);
                    break; // Stop searching once a suitable VM is found
                }
            }
            // If no suitable VM was found after traversing all datacenters
            if (targetVm == Vm.NULL) {
                jobsToProcessList.remove(selectedCloudlet);
            }
        }

    }

    private void executeEarliestShortestCloudletToMostFreeDcAction() {
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
                    bind(selectedCloudlet, targetVm);
                    jobsToProcessList.remove(selectedCloudlet);

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

    }

    // this action is if the agent performs cloudlet to DC mapping
    private void executeRlCloudletToDcAction(final int[] action) {
        final double targetTime = proxy().calculateTargetTime();
        // action[i] refers to observation slot i, i.e. the i-th visible job.
        final List<Cloudlet> visibleJobs = proxy().getVisibleJobs(targetTime);
        if (action.length < visibleJobs.size()) {
            throw new IllegalArgumentException("Got " + action.length + " actions for "
                    + visibleJobs.size() + " visible jobs; max_jobs_waiting must match the action space");
        }

        for (int i = 0; i < visibleJobs.size(); i++) {
            final CloudletWithLocation job = (CloudletWithLocation) visibleJobs.get(i);
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
            bind(job, vm);
        }
    }

    private double calculateJobsPlacedRatio(final int jobsPlaced, final int jobsWaiting) {
        if (jobsWaiting == 0) {
            return 0.0;
        }
        return (double) jobsPlaced / jobsWaiting;
    }

    /**
     * Per host, in datacenter order: [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts].
     * <p>
     * dc_id is the action index of the host's datacenter (CloudSim ids start at 2, action 0 is the
     * no-op, so dc_id = id - 1). vm_capacity_pes is the PE count of the host's VMs, which decides
     * whether a job can be held at all. free_pes is clipped at 0 once cloudlets queue, and
     * backlog_core_ts carries the depth that clipping hides: the core-timesteps of work still to
     * run on the host's VMs, executing, waiting, and still crossing the network to them.
     */
    private int[] getInfraObsPerHost() {
        final int totalHosts = getTotalHosts();
        final int[] infrastructureObservation = new int[HOST_OBS_FEATURES * totalHosts];
        final double interval = simSettings.getTimestepInterval();
        List<Datacenter> datacenterList = proxy().getSimulation().getCis().getDatacenterList();
        int currentIndex = 0;
        for (Datacenter dc : datacenterList) {
            for (Host host : dc.getHostList()) {
                long vmCapacityPes = 0;
                long usedPes = 0;
                double backlogCoreSeconds = 0;
                for (Vm vm : host.getVmList()) {
                    final CloudletScheduler scheduler = vm.getCloudletScheduler();
                    vmCapacityPes += vm.getPesNumber();
                    usedPes += scheduler.getCloudletList().stream()
                            .mapToLong(Cloudlet::getPesNumber).sum();
                    final double now = clock();
                    backlogCoreSeconds += scheduler.getCloudletExecList().stream()
                            .mapToDouble(ce -> ce.getCloudlet().getPesNumber()
                                    * remainingLength(ce, vm, now) / vm.getMips())
                            .sum();
                    backlogCoreSeconds += scheduler.getCloudletWaitingList().stream()
                            .mapToDouble(ce -> ce.getCloudlet().getPesNumber()
                                    * ce.getCloudlet().getLength() / vm.getMips())
                            .sum();
                    for (Cloudlet inFlight : proxy().getInFlight(vm).keySet()) {
                        usedPes += inFlight.getPesNumber();
                        backlogCoreSeconds +=
                                inFlight.getPesNumber() * inFlight.getLength() / vm.getMips();
                    }
                }
                infrastructureObservation[currentIndex++] = (int) dc.getId() - 1;
                infrastructureObservation[currentIndex++] =
                        getDcTypeIdFromStr(((DatacenterWithType) dc).getType());
                infrastructureObservation[currentIndex++] = (int) vmCapacityPes;
                infrastructureObservation[currentIndex++] =
                        (int) Math.max(0, vmCapacityPes - usedPes);
                infrastructureObservation[currentIndex++] =
                        (int) Math.ceil(backlogCoreSeconds / interval);
            }
        }
        return infrastructureObservation;
    }

    /**
     * Per-PE MI a running cloudlet still has to execute at time now. CloudSim advances a
     * cloudlet's progress only when its datacenter processes an event, which with no scheduling
     * interval happens at the next finish in that DC, so its own counter can be many timesteps
     * stale. Space-shared with full utilisation, a cloudlet runs at the VM's per-PE MIPS from
     * the moment it starts, so read the progress from the clock instead.
     */
    private static double remainingLength(final CloudletExecution ce, final Vm vm,
            final double now) {
        final Cloudlet cloudlet = ce.getCloudlet();
        return Math.max(0, cloudlet.getLength() - (now - cloudlet.getStartTime()) * vm.getMips());
    }

    // 0 is reserved for padding host slots, so real types start at 1.
    private int getDcTypeIdFromStr(final String dcType) {
        return switch (dcType) {
            case "cloud" -> 1;
            case "edge" -> 2;
            case "micro" -> 3;
            default -> throw new IllegalArgumentException("Unexpected DC type: " + dcType);
        };
    }

    // ============== Shaping potential ==============

    private double potential() {
        final double now = clock();
        final Map<Cloudlet, Double> placed = estimatePlacedCompletionTimes(now);
        return ledger.potential(now, job -> placed.containsKey(job) ? placed.get(job)
                : bestCaseCompletionTime(job, now));
    }

    /**
     * Completion time of every job placed on a VM, replaying each VM's space-shared FIFO:
     * executing jobs hold their PEs until they finish, then waiting and in-flight jobs start
     * in order, each when enough PEs are free and it has arrived.
     */
    private Map<Cloudlet, Double> estimatePlacedCompletionTimes(final double now) {
        final Map<Cloudlet, Double> completion = new HashMap<>();
        for (Vm vm : proxy().getBroker().getVmExecList()) {
            final CloudletScheduler scheduler = vm.getCloudletScheduler();
            final double mips = vm.getMips();
            final PriorityQueue<Double> peFreeAt = new PriorityQueue<>();
            for (int pe = 0; pe < vm.getPesNumber(); pe++) {
                peFreeAt.add(now);
            }
            for (CloudletExecution ce : scheduler.getCloudletExecList()) {
                final long pes = ce.getCloudlet().getPesNumber();
                final double finish = takePes(peFreeAt, pes, now)
                        + remainingLength(ce, vm, now) / mips;
                releasePes(peFreeAt, pes, finish);
                completion.put(ce.getCloudlet(), finish);
            }
            final Map<Cloudlet, Double> queued = new LinkedHashMap<>();
            scheduler.getCloudletWaitingList().forEach(ce -> queued.put(ce.getCloudlet(), now));
            queued.putAll(proxy().getInFlight(vm));
            queued.forEach((cloudlet, arrival) -> {
                final long pes = cloudlet.getPesNumber();
                final double finish = takePes(peFreeAt, pes, arrival) + cloudlet.getLength() / mips;
                releasePes(peFreeAt, pes, finish);
                completion.put(cloudlet, finish);
            });
        }
        return completion;
    }

    /** Takes the pes earliest-free PEs: the job starts once all are free and it has arrived. */
    private static double takePes(final PriorityQueue<Double> peFreeAt, final long pes,
            final double arrival) {
        double start = arrival;
        for (long i = 0; i < pes; i++) {
            start = Math.max(start, peFreeAt.poll());
        }
        return start;
    }

    private static void releasePes(final PriorityQueue<Double> peFreeAt, final long pes,
            final double at) {
        for (long i = 0; i < pes; i++) {
            peFreeAt.add(at);
        }
    }

    /** An unplaced job's completion time if it went now to the fastest DC it may use, idle. */
    private double bestCaseCompletionTime(final Cloudlet job, final double now) {
        double best = Double.POSITIVE_INFINITY;
        for (int dc : reachableDcsByLocation.get(((CloudletWithLocation) job).getLocation())) {
            final String dcType = ((DatacenterWithType) proxy().getDatacenterByIdx(dc)).getType();
            best = Math.min(best, simSettings.networkDelay(dcType) + job.getLength() / dcPeMips[dc]);
        }
        return now + best;
    }

    /** Per DC index: fastest VM PE, and which DCs a job from that origin may use. */
    @SuppressWarnings("unchecked")
    private void cacheTopology() {
        final List<Map<String, Object>> dcMaps = simSettings.getDatacenters();
        dcPeMips = new double[dcMaps.size()];
        reachableDcsByLocation = new ArrayList<>();
        for (int dc = 0; dc < dcMaps.size(); dc++) {
            for (Map<String, Object> host : (List<Map<String, Object>>) dcMaps.get(dc).get("hosts")) {
                for (Map<String, Object> vm : (List<Map<String, Object>>) host.get("vms")) {
                    dcPeMips[dc] = Math.max(dcPeMips[dc], ((Number) vm.get("pe_mips")).doubleValue());
                }
            }
            final List<Integer> connectTo =
                    ((DatacenterWithType) proxy().getDatacenterByIdx(dc)).getConnectTo();
            final List<Integer> reachable = new ArrayList<>();
            if (connectTo.isEmpty()) {
                for (int other = 0; other < dcMaps.size(); other++) {
                    reachable.add(other);
                }
            } else {
                reachable.add(dc);
                reachable.addAll(connectTo);
            }
            reachableDcsByLocation.add(reachable);
        }
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

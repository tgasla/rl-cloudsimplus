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
import java.util.Optional;
import java.util.stream.IntStream;


public class WrappedSimulation extends WrappedSimulationBase {

    // Must match Python JobPlacementEnv.HOST_OBS_FEATURES:
    // [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts]
    static final int HOST_OBS_FEATURES = 5;
    private static final int DC_ID_COL = 0;
    private static final int FREE_PES_COL = 3;
    private static final int BACKLOG_COL = 4;

    // Concrete settings reference for domain-specific access
    private final SimulationSettings simSettings;

    // RL-level episode tracking (RL concepts; not present in CloudSimProxy)
    private double bestEpisodeReward;
    private double currentEpisodeReward;
    private double lastReward = 0.0;

    // Net-SLA accounting of the current episode, and the shaping potential of the last state
    private SlaLedger ledger;
    private double lastPotential;
    // Per DC index: the fastest PE, the largest VM, and the job origins (by DC index) each DC can serve
    private double[] dcPeMips;
    private long[] maxVmPes;
    private List<List<Integer>> reachableDcsByLocation;
    // Ties between DC indices in the rule-based policies: tier, capacity, name
    private Comparator<Integer> canonicalDcOrder;

    // Action-phase placement counter: set by bind(), consumed in step info
    private int jobsPlacedThisTimestep;
    // PEs bound to each VM this step, not yet submitted to it (for the VM selector)
    private final Map<Vm, Long> pesBoundThisTimestep = new HashMap<>();

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
        // A job arriving at or after the horizon could never be placed: it would only be charged.
        final double horizon = simSettings.getMaxEpisodeLength() * simSettings.getTimestepInterval();
        final double lastArrival = proxy().jobArrivalTimeMap.values().stream()
                .mapToDouble(Double::doubleValue).max().orElse(0);
        if (lastArrival >= horizon) {
            throw new IllegalArgumentException("a job arrives at t=" + lastArrival
                    + ", at or after the horizon t=" + horizon + " (max_episode_length)");
        }
        ledger = new SlaLedger(proxy().getSimulationCloudletList(), proxy().jobArrivalTimeMap,
                simSettings);
        cacheTopology();
        lastPotential = simSettings.isRewardShaping() ? potential() : 0;
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
        pesBoundThisTimestep.clear();
        executeCustomCloudletToDcAction(action);

        final double targetTime = proxy.calculateTargetTime();
        final int jobsWaiting = proxy.getJobsToSubmitAtThisTimestep(targetTime).size();

        proxy.runOneTimestep();
        proxy.pruneInFlight();
        ledger.beginStep();
        proxy.evict(ledger.resolve(clock()));
        if (currentStep >= simSettings.getMaxEpisodeLength()) {
            drainToResolution();
        }

        // Every job resolves, so the episode ends when none is left, at the latest at the horizon.
        final boolean terminated = !ledger.anyUnresolved();
        final double unshapedReward = ledger.stepReward();
        // The potential is only needed, and only computed, when shaping is on.
        final double potential = terminated || !simSettings.isRewardShaping() ? 0 : potential();
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
            proxy().pruneInFlight();
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
        pesBoundThisTimestep.merge(vm, job.getPesNumber(), Long::sum);
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
        for (Vm vm : proxy().getBroker().getVmExecList()) {
            final int dcId = (int) vm.getHost().getDatacenter().getId();
            if (dcId != targetDcId) {
                continue;
            }
            final long expectedFreePes = expectedFreePes(vm);

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

    /**
     * PEs of vm not taken by jobs executing or queued on it, crossing the network to it, or
     * bound to it this step. Negative once jobs queue: it then measures how overloaded it is.
     */
    private long expectedFreePes(final Vm vm) {
        final long usedVmPes = vm.getCloudletScheduler().getCloudletList().stream()
                .mapToLong(Cloudlet::getPesNumber).sum()
                + proxy().getInFlight(vm).keySet().stream()
                        .mapToLong(Cloudlet::getPesNumber).sum();
        return vm.getPesNumber() - usedVmPes - pesBoundThisTimestep.getOrDefault(vm, 0L);
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

    // The two rule-based policies decide as the agent does: on its window (the visible jobs), its
    // observation (each host's free_pes, clipped at 0, and backlog_core_ts) and its action mask
    // (legalDcs), and they place through its action path, one DC per slot. A job a rule places
    // counts as using its PEs on the DC's host with the most free PEs (the first such host) for
    // the rest of the step. benchmark/reference_policies.py R1/R2 are the same two rules, played
    // from the Python side of the observation.

    /**
     * R2: the job due first (then the most critical) goes to the nearest DC it may use with a
     * host that has the PEs free: its own DC, then its neighbours, then the cloud, ties by tier,
     * capacity and name. A job that finds none waits and is tried again next step, until it is
     * placed or evicted at its due time.
     */
    private void executeEarliestMostCriticalCloudletToNearestDcAction() {
        final List<Cloudlet> window = proxy().getVisibleJobs(proxy().calculateTargetTime());
        final List<List<long[]>> hosts = observedHosts();
        final int[] action = new int[window.size()];
        for (int slot : slotsInOrder(window, Comparator.comparingDouble((Cloudlet c) -> proxy().getDueTime(c))
                .thenComparingInt(c -> -((CloudletWithLocation) c).getDelaySensitivity())
                .thenComparingDouble(c -> proxy().jobArrivalTimeMap.get(c.getId()))
                .thenComparingLong(Cloudlet::getId))) {
            final Cloudlet job = window.get(slot);
            final List<Integer> nearestFirst = legalDcs(job).stream()
                    .sorted(Comparator.comparingInt((Integer dc) -> hops(job, dc))
                            .thenComparing(canonicalDcOrder))
                    .toList();
            for (int dc : nearestFirst) {
                final long[] host = hosts.get(dc).get(mostFreeHost(hosts.get(dc)));
                if (host[FREE_PES_COL] >= job.getPesNumber()) {
                    host[FREE_PES_COL] -= job.getPesNumber();
                    action[slot] = dc + 1;
                    break;
                }
            }
        }
        executeRlCloudletToDcAction(action);
    }

    /**
     * R1: the job due first (then the shortest) goes to the DC it may use with the most free PEs
     * (the sum of its hosts' free_pes), then the least backlog (the sum of its hosts'
     * backlog_core_ts plus the core-timesteps placed there this step, at the host's speed), then
     * by tier, capacity and name. It always places: a full DC queues the job.
     */
    private void executeEarliestShortestCloudletToMostFreeDcAction() {
        final List<Cloudlet> window = proxy().getVisibleJobs(proxy().calculateTargetTime());
        final List<List<long[]>> hosts = observedHosts();
        final double[] placedWork = new double[hosts.size()];
        final int[] action = new int[window.size()];
        for (int slot : slotsInOrder(window, Comparator.comparingDouble((Cloudlet c) -> proxy().getDueTime(c))
                .thenComparingLong(Cloudlet::getLength)
                .thenComparingDouble(c -> proxy().jobArrivalTimeMap.get(c.getId()))
                .thenComparingLong(Cloudlet::getId))) {
            final Cloudlet job = window.get(slot);
            final Optional<Integer> mostFree = legalDcs(job).stream()
                    .min(Comparator.comparingLong((Integer dc) -> -sum(hosts.get(dc), FREE_PES_COL))
                            .thenComparingDouble(dc -> sum(hosts.get(dc), BACKLOG_COL) + placedWork[dc])
                            .thenComparing(canonicalDcOrder));
            if (mostFree.isEmpty()) {
                continue;
            }
            final int dc = mostFree.get();
            final int h = mostFreeHost(hosts.get(dc));
            final long[] host = hosts.get(dc).get(h);
            host[FREE_PES_COL] = Math.max(0, host[FREE_PES_COL] - job.getPesNumber());
            placedWork[dc] += job.getPesNumber() * proxy().nominalRuntime(job) * simSettings.getMipsRef()
                    / hostMips(dc, h);
            action[slot] = dc + 1;
        }
        executeRlCloudletToDcAction(action);
    }

    /** The infrastructure observation's host rows, by DC index, in host order. */
    private List<List<long[]>> observedHosts() {
        final int[] obs = getInfraObsPerHost();
        final List<List<long[]>> byDc = new ArrayList<>();
        for (int dc = 0; dc < maxVmPes.length; dc++) {
            byDc.add(new ArrayList<>());
        }
        for (int row = 0; row < obs.length; row += HOST_OBS_FEATURES) {
            final long[] host = Arrays.stream(obs, row, row + HOST_OBS_FEATURES).asLongStream().toArray();
            byDc.get((int) host[DC_ID_COL] - 1).add(host);    // dc_id is the DC index + 1
        }
        return byDc;
    }

    private static long sum(final List<long[]> hosts, final int column) {
        long total = 0;
        for (long[] host : hosts) {
            total += host[column];
        }
        return total;
    }

    /** The first host with the most free PEs. */
    private static int mostFreeHost(final List<long[]> hosts) {
        int best = 0;
        for (int h = 1; h < hosts.size(); h++) {
            if (hosts.get(h)[FREE_PES_COL] > hosts.get(best)[FREE_PES_COL]) {
                best = h;
            }
        }
        return best;
    }

    /** Per-PE MIPS of host h of DC index dc (one VM per host, as JobPlacementEnv requires). */
    private double hostMips(final int dc, final int h) {
        return proxy().getDatacenterByIdx(dc).getHostList().get(h).getVmList().get(0).getMips();
    }

    /** Slot indices of the window, ordered by their jobs. */
    private static List<Integer> slotsInOrder(final List<Cloudlet> window, final Comparator<Cloudlet> order) {
        return IntStream.range(0, window.size()).boxed()
                .sorted(Comparator.comparing(window::get, order)).toList();
    }

    /** DC indices a job may be placed on: its origin and the DCs the origin connects to. */
    private List<Integer> reachableDcs(final Cloudlet job) {
        return reachableDcsByLocation.get(((CloudletWithLocation) job).getLocation());
    }

    /** The DCs the agent's action mask allows for a job: within reach, with a VM that can hold it. */
    private List<Integer> legalDcs(final Cloudlet job) {
        return reachableDcs(job).stream().filter(dc -> maxVmPes[dc] >= job.getPesNumber()).toList();
    }

    /** Hops from the job's origin to DC index dc: 0 itself, 1 a neighbour, 2 the cloud. */
    private int hops(final Cloudlet job, final int dc) {
        if (dc == ((CloudletWithLocation) job).getLocation()) {
            return 0;
        }
        return "cloud".equals(((DatacenterWithType) proxy().getDatacenterByIdx(dc)).getType()) ? 2 : 1;
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
     * Completion time of every job placed on a VM, replaying each VM's space-shared scheduler:
     * a running job completes when its execution ends (SlaLedger.executionEnd, as the ledger
     * judges it) and holds its PEs until then, or until now if CloudSim has yet to record it;
     * whenever PEs free up or a job arrives, the queued jobs are scanned in order (waiting list,
     * then in-flight jobs by arrival) and every one that has arrived and fits starts, as CloudSim
     * moves waiting cloudlets to execution.
     */
    Map<Cloudlet, Double> estimatePlacedCompletionTimes(final double now) {
        final Map<Cloudlet, Double> completion = new HashMap<>();
        for (Vm vm : proxy().getBroker().getVmExecList()) {
            final CloudletScheduler scheduler = vm.getCloudletScheduler();
            final double mips = vm.getMips();
            final PriorityQueue<double[]> releases =                        // {time, pes}
                    new PriorityQueue<>(Comparator.comparingDouble(r -> r[0]));
            long freePes = vm.getPesNumber();
            for (CloudletExecution ce : scheduler.getCloudletExecList()) {
                final double end = SlaLedger.executionEnd(ce.getCloudlet());
                completion.put(ce.getCloudlet(), end);
                releases.add(new double[] {Math.max(now, end), ce.getCloudlet().getPesNumber()});
                freePes -= ce.getCloudlet().getPesNumber();
            }
            final Map<Cloudlet, Double> queued = new LinkedHashMap<>();       // job -> arrival
            scheduler.getCloudletWaitingList().forEach(ce -> queued.put(ce.getCloudlet(), now));
            proxy().getInFlight(vm).entrySet().stream()
                    .sorted(Map.Entry.comparingByValue())
                    .forEach(e -> queued.put(e.getKey(), e.getValue()));
            double t = now;
            while (!queued.isEmpty()) {
                for (Iterator<Map.Entry<Cloudlet, Double>> it = queued.entrySet().iterator(); it.hasNext();) {
                    final Map.Entry<Cloudlet, Double> e = it.next();
                    final long pes = e.getKey().getPesNumber();
                    if (e.getValue() <= t && pes <= freePes) {
                        final double finish = t + e.getKey().getLength() / mips;
                        completion.put(e.getKey(), finish);
                        releases.add(new double[] {finish, pes});
                        freePes -= pes;
                        it.remove();
                    }
                }
                final double at = t;
                final double nextArrival = queued.values().stream()
                        .filter(a -> a > at).mapToDouble(Double::doubleValue).min()
                        .orElse(Double.POSITIVE_INFINITY);
                final double nextRelease = releases.isEmpty() ? Double.POSITIVE_INFINITY
                        : releases.peek()[0];
                if (Double.isInfinite(nextArrival) && Double.isInfinite(nextRelease)) {
                    break; // a queued job larger than its VM; the selector never binds one
                }
                t = Math.min(nextArrival, nextRelease);
                while (!releases.isEmpty() && releases.peek()[0] <= t) {
                    freePes += (long) releases.poll()[1];
                }
            }
        }
        return completion;
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

    /** Per DC index: fastest VM PE, largest VM, and which DCs a job from that origin may use. */
    @SuppressWarnings("unchecked")
    private void cacheTopology() {
        final List<Map<String, Object>> dcMaps = simSettings.getDatacenters();
        dcPeMips = new double[dcMaps.size()];
        maxVmPes = new long[dcMaps.size()];
        final long[] capacityPes = new long[dcMaps.size()];
        reachableDcsByLocation = new ArrayList<>();
        for (int dc = 0; dc < dcMaps.size(); dc++) {
            for (Map<String, Object> host : (List<Map<String, Object>>) dcMaps.get(dc).get("hosts")) {
                for (Map<String, Object> vm : (List<Map<String, Object>>) host.get("vms")) {
                    dcPeMips[dc] = Math.max(dcPeMips[dc], CloudSimProxy.scaledMips(vm.get("pe_mips")));
                    maxVmPes[dc] = Math.max(maxVmPes[dc], ((Number) vm.get("pes")).longValue());
                    capacityPes[dc] += ((Number) host.get("amount")).longValue()
                            * ((Number) vm.get("amount")).longValue() * ((Number) vm.get("pes")).longValue();
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
        // Physical attributes only, never the DC's position, so renumbering DCs changes no choice.
        canonicalDcOrder = Comparator
                .comparingInt((Integer dc) -> getDcTypeIdFromStr((String) dcMaps.get(dc).get("type")))
                .thenComparingLong(dc -> -capacityPes[dc])
                .thenComparing(dc -> (String) dcMaps.get(dc).get("name"));
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

package daislab.cspg;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.listeners.EventListener;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class CloudSimProxyBase implements ICloudSimProxy {

    protected final Logger LOGGER = LoggerFactory.getLogger(getClass().getSimpleName());
    protected final ISimulationSettings settings;
    protected final CloudSimPlus cloudSimPlus;
    protected final DatacenterBrokerFirstFitFixed broker;
    protected final List<Cloudlet> inputJobs;
    protected final PriorityQueue<Cloudlet> jobQueue;
    protected final Map<Long, Double> jobArrivalTimeMap;
    protected List<Double> jobsFinishedWaitTimeLastTimestep;
    protected int vmsCreated;
    protected boolean firstStep;

    protected CloudSimProxyBase(final ISimulationSettings settings,
            final List<Cloudlet> inputJobs) {
        this.settings = settings;
        this.inputJobs = new ArrayList<>(inputJobs);
        final int initialCapacity = Math.max(inputJobs.size(), 1);
        jobQueue = new PriorityQueue<>(initialCapacity,
                (c1, c2) -> Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        jobQueue.addAll(this.inputJobs);
        jobArrivalTimeMap = jobQueue.stream()
                .collect(Collectors.toMap(Cloudlet::getId, Cloudlet::getSubmissionDelay));
        cloudSimPlus = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        broker = new DatacenterBrokerFirstFitFixed(cloudSimPlus);
        broker.setShutdownWhenIdle(false);
        jobsFinishedWaitTimeLastTimestep = new ArrayList<>();
        vmsCreated = 0;
        firstStep = true;

        setupInfrastructure();
        initializeJobListeners();
        ensureAllJobsCompleteBeforeSimulationEnds();
        cloudSimPlus.startSync();
        proceedClockTo(settings.getMinTimeBetweenEvents());
    }

    /** Domain creates datacenters and submits initial VMs. */
    protected abstract void setupInfrastructure();

    /** Returns the primary datacenter used for keepalive events. */
    protected abstract Datacenter getPrimaryDatacenter();

    /** Domain-specific job submission filter logic. */
    protected abstract void tryToSubmitJobs(List<Cloudlet> cloudletList);

    /** Domain-specific stats printing called each timestep. */
    protected abstract void printStats();

    // ============== Simulation stepping ==============

    private void proceedClockTo(final double targetTime) {
        double adjustedInterval = targetTime - clock();
        int maxIterations = 1000;
        int iterations = 0;

        LOGGER.info("{}: Proceeding clock to {}", clock(), targetTime);

        while (cloudSimPlus.runFor(adjustedInterval) < targetTime) {
            adjustedInterval = targetTime - clock();
            adjustedInterval =
                    adjustedInterval <= 0 ? settings.getMinTimeBetweenEvents() : adjustedInterval;

            if (++iterations >= maxIterations) {
                LOGGER.warn(
                        "Exceeded maximum iterations in runForInternal. Breaking the loop to prevent infinite loop.");
                break;
            }
        }
        LOGGER.info("Current clock time is {}", clock());
    }

    double calculateTargetTime() {
        if (firstStep) {
            return settings.getTimestepInterval();
        }
        return clock() + settings.getTimestepInterval();
    }

    private double calculateStartTime() {
        if (firstStep) {
            return settings.getMinTimeBetweenEvents();
        }
        return Math.max(clock() - settings.getTimestepInterval(), 0);
    }

    @Override
    public void runOneTimestep() {
        final double targetTime = calculateTargetTime();
        ensureSimulationIsRunning();
        jobsFinishedWaitTimeLastTimestep.clear();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        maybeClearLists();
        tryToSubmitJobs(jobsToSubmitList);
        proceedClockTo(targetTime);
        if (settings.isPrintStats()) {
            printStats();
        }
        if (firstStep) {
            firstStep = false;
        }
    }

    private void ensureSimulationIsRunning() {
        if (!isRunning()) {
            throw new IllegalStateException(
                    "Simulation is not running. Please reset or create a new one!");
        }
    }

    private void maybeClearLists() {
        if (settings.isClearCreatedLists()) {
            broker.getCloudletCreatedList().clear();
            // broker.getVmCreatedList().clear(); // uncomment to also purge the VM created list
        }
    }

    // ============== Job/event management ==============

    private void initializeJobListeners() {
        inputJobs.forEach(this::addOnStartListener);
        inputJobs.forEach(this::addOnFinishListener);
    }

    private void ensureAllJobsCompleteBeforeSimulationEnds() {
        double interval = settings.getTimestepInterval();
        cloudSimPlus.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && hasUnfinishedJobs()) {
                LOGGER.trace("Jobs not finished. Sending empty event to keep simulation running.");
                Datacenter dc = getPrimaryDatacenter();
                cloudSimPlus.send(dc, dc, interval, CloudSimTag.NONE, null);
            }
        });
    }

    private void addOnStartListener(Cloudlet cloudlet) {
        cloudlet.addOnStartListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug(
                        "{}: Cloudlet {} started on VM {}, Host {}, DC {}. Free cores: {}, expected free: {}",
                        clock(), cloudlet.getId(), cloudlet.getVm().getId(),
                        cloudlet.getVm().getHost().getId(),
                        cloudlet.getVm().getHost().getDatacenter().getId(),
                        cloudlet.getVm().getFreePesNumber(),
                        cloudlet.getVm().getExpectedFreePesNumber());
            }
        });
    }

    private void addOnFinishListener(Cloudlet cloudlet) {
        cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                final double waitTime =
                        cloudlet.getStartTime() - jobArrivalTimeMap.get(cloudlet.getId());
                jobsFinishedWaitTimeLastTimestep.add(waitTime);
                LOGGER.debug(
                        "{}: Cloudlet {}, {} mi, {} cores on vm{}/host{}/dc{}. "
                                + "Arrived {}, started {}, finished {}, exec {}, waited {}",
                        clock(), cloudlet.getId(), cloudlet.getLength(), cloudlet.getPesNumber(),
                        cloudlet.getVm().getId(), cloudlet.getVm().getHost().getId(),
                        cloudlet.getVm().getHost().getDatacenter().getId(),
                        jobArrivalTimeMap.get(cloudlet.getId()), cloudlet.getStartTime(),
                        cloudlet.getFinishTime(), cloudlet.getTotalExecutionTime(), waitTime);
            }
        });
    }

    List<Cloudlet> getJobsToSubmitAtThisTimestep(final double targetTime) {
        return jobQueue.stream().takeWhile(cloudlet -> cloudlet.getSubmissionDelay() < targetTime)
                .collect(Collectors.toList());
    }

    protected boolean isAnyVmSuitableForCloudlet(Cloudlet cloudlet) {
        return broker.getVmExecList().stream().anyMatch(vm -> vm.isSuitableForCloudlet(cloudlet));
    }

    protected void submitCloudletList(final List<Cloudlet> cloudlets) {
        broker.submitCloudletList(cloudlets);
    }

    // ============== Status / stats ==============

    public void printCloudletStatus() {
        final double startTime = calculateStartTime();
        LOGGER.info("[{} - {}): All jobs: {}", startTime, clock(), inputJobs.size());
        Map<Cloudlet.Status, List<Long>> jobsByStatus = new HashMap<>();
        for (Cloudlet c : inputJobs) {
            jobsByStatus.computeIfAbsent(c.getStatus(), k -> new ArrayList<>()).add(c.getId());
        }
        for (Map.Entry<Cloudlet.Status, List<Long>> e : jobsByStatus.entrySet()) {
            LOGGER.debug("[{} - {}): {}: {} ({})", startTime, clock(), e.getKey(),
                    e.getValue().size(), e.getValue());
        }
        LOGGER.info("[{} - {}): ARRIVED: {}", startTime, clock(), getArrivedJobsCount());
    }

    // ============== ICloudSimProxy ==============

    @Override
    public boolean isRunning() {
        return cloudSimPlus.isRunning() && hasUnfinishedJobs();
    }

    @Override
    public void terminate() {
        cloudSimPlus.terminate();
    }

    @Override
    public long getNumberOfFutureEvents() {
        return cloudSimPlus.getNumberOfFutureEvents(simEvent -> true);
    }

    @Override
    public double clock() {
        return cloudSimPlus.clock();
    }

    @Override
    public DatacenterBroker getBroker() {
        return broker;
    }

    @Override
    public long getFinishedJobsCount() {
        return inputJobs.parallelStream().filter(c -> c.getStatus().equals(Cloudlet.Status.SUCCESS))
                .count();
    }

    // ============== Shared public accessors ==============

    public CloudSimPlus getSimulation() {
        return cloudSimPlus;
    }

    public long getArrivedJobsCount() {
        return jobArrivalTimeMap.entrySet().parallelStream()
                .filter(entry -> entry.getValue() < clock()).count();
    }

    public List<Double> getFinishedJobsWaitTimeLastTimestep() {
        return jobsFinishedWaitTimeLastTimestep;
    }

    public long getNotYetRunningJobsCount() {
        return inputJobs.parallelStream().filter(c -> jobArrivalTimeMap.get(c.getId()) < clock())
                .filter(c -> !c.getStatus().equals(Cloudlet.Status.INEXEC))
                .filter(c -> !c.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

    public long getRunningJobsCount() {
        return inputJobs.parallelStream().filter(c -> c.getStatus().equals(Cloudlet.Status.INEXEC))
                .count();
    }

    public long getAllocatedCores() {
        return broker.getVmExecList().parallelStream().map(Vm::getPesNumber).reduce(Long::sum)
                .orElse(0L);
    }

    public double[] getVmCpuUsage() {
        List<Vm> vms = broker.getVmExecList();
        double[] usage = new double[vms.size()];
        IntStream.range(0, vms.size())
                .forEach(i -> usage[i] = vms.get(i).getCpuPercentUtilization());
        return usage;
    }

    public double[] getVmMemoryUsage() {
        List<Vm> vms = broker.getVmExecList();
        double[] usage = new double[vms.size()];
        IntStream.range(0, vms.size())
                .forEach(i -> usage[i] = vms.get(i).getRam().getPercentUtilization());
        return usage;
    }

    public int getLastCreatedVmId() {
        return vmsCreated - 1;
    }

    // ============== Shared domain helpers ==============

    long calculateMaxJobCoresNeeded() {
        return getJobsToSubmitAtThisTimestep(calculateTargetTime()).stream()
                .mapToLong(Cloudlet::getPesNumber).max().orElse(0);
    }

    /** Shared utility: builds a PE list for host or VM creation. Both domains use this. */
    protected static List<Pe> createPeList(long pes, long mips) {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < pes; i++) {
            peList.add(new PeSimple(mips, new PeProvisionerSimple()));
        }
        return peList;
    }

    // ============== Private helpers ==============

    private boolean hasUnfinishedJobs() {
        return broker.getCloudletFinishedList().size() < inputJobs.size();
    }
}

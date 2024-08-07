package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostAbstract;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.listeners.DatacenterBrokerEventInfo;
import org.cloudsimplus.listeners.EventListener;
import org.cloudsimplus.resources.Ram;
import org.cloudsimplus.resources.Bandwidth;
import org.cloudsimplus.resources.ResourceManageable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.IntStream;

public class CloudSimProxy {

    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};

    private static final Logger LOGGER =
        LoggerFactory.getLogger(CloudSimProxy.class.getSimpleName());
    private static final double minTimeBetweenEvents = 0.1;

    private final CloudSimPlus cloudSimPlus;
    private final Datacenter datacenter;
    private final DatacenterBrokerFirstFitFixed broker;
    private final SimulationSettings settings;
    private final VmCost vmCost;
    private final Map<Long, Double> originalSubmissionDelay;
    private final List<Cloudlet> inputJobs;
    private final List<Cloudlet> unsubmittedJobs;
    private int unableToSubmitJobCount;
    private List<Double> jobsFinishedWaitTimeLastInterval;
    private int nextVmId;
    private int episodeCount;
    private int lastSubmittedJobIndex;
    private int previousLastSubmittedJobIndex;

    public CloudSimProxy(
        final SimulationSettings settings,
        final List<Cloudlet> inputJobs,
        final int episodeCount
) {
        this.inputJobs = new ArrayList<>(inputJobs);
        this.unsubmittedJobs = new ArrayList<>(inputJobs);
        originalSubmissionDelay = new HashMap<>();
        this.settings = settings;
        this.episodeCount = episodeCount;
        nextVmId = 0;
        lastSubmittedJobIndex = 0;
        previousLastSubmittedJobIndex = 0;
        unableToSubmitJobCount = 0;
        jobsFinishedWaitTimeLastInterval = new ArrayList<>();

        cloudSimPlus = new CloudSimPlus(minTimeBetweenEvents);
        broker = new DatacenterBrokerFirstFitFixed(cloudSimPlus);
        datacenter = createDatacenter();
        vmCost = new VmCost(
            settings.getVmRunningHourlyCost(),
            settings.getTimestepInterval(),
            settings.isPayingForTheFullHour());

        final List<Vm> smallVmList = createVmList(settings.getInitialSVmCount(), SMALL);
        final List<Vm> mediumVmList = createVmList(settings.getInitialMVmCount(), MEDIUM);
        final List<Vm> largeVmList = createVmList(settings.getInitialLVmCount(), LARGE);
        broker.submitVmList(smallVmList);
        broker.submitVmList(mediumVmList);
        broker.submitVmList(largeVmList);

        Collections.sort(this.inputJobs, (c1, c2) ->
            Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        this.inputJobs.forEach(c -> originalSubmissionDelay.put(c.getId(), c.getSubmissionDelay()));
        this.inputJobs.forEach(c -> addOnStartListener(c));
        this.inputJobs.forEach(c -> addOnFinishListener(c));

        ensureAllJobsCompleteBeforeSimulationEnds();

        cloudSimPlus.startSync();
        runFor(minTimeBetweenEvents);
    }

    public static int getSizeMultiplier(final String type) {
        int sizeMultiplier;

        switch (type) {
            case MEDIUM:
                sizeMultiplier = 2; // m5a.xlarge
                break;
            case LARGE:
                sizeMultiplier = 4; // m5a.2xlarge
                break;
            case SMALL:
            default:
                sizeMultiplier = 1; // m5a.large
        }
        return sizeMultiplier;
    }

    public long getVmCoreCountByType(final String type) {
        return settings.getBasicVmPeCnt() * getSizeMultiplier(type);
    }

    private void ensureAllJobsCompleteBeforeSimulationEnds() {
        cloudSimPlus.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && hasUnfinishedJobs()) {
                LOGGER.debug("There are unfinished jobs. "
                    + "Sending a NONE event to prevent simulation from ending.");
                // this will prevent the simulation from ending
                // while some jobs have not yet finished running
                sendEmptyEventAt(settings.getTimestepInterval());
            }
        });
    }

    private void sendEmptyEventAt(double time) {
        // We add NONE events to prevent the simulation from ending
            cloudSimPlus.send(
                datacenter,
                datacenter,
                time,
                CloudSimTag.NONE,
                null
            );
    }

    private Datacenter createDatacenter() {
        List<Host> hostList = new ArrayList<>();

        for (int i = 0; i < settings.getDatacenterHostsCnt(); i++) {
            List<Pe> peList = createPeList();

            final long hostRam = settings.getHostRam();
            final long hostBw = settings.getHostBw();
            final long hostSize = settings.getHostSize();
            Host host =
                new HostWithoutCreatedList(hostRam, hostBw, hostSize, peList)
                    .setRamProvisioner(new ResourceProvisionerSimple())
                    .setBwProvisioner(new ResourceProvisionerSimple())
                    .setVmScheduler(new VmSchedulerTimeShared());

            hostList.add(host);
        }
        LOGGER.debug("Creating datacenter...");
        return new LoggingDatacenter(cloudSimPlus, hostList, new VmAllocationPolicyRl());
    }

    private List<Vm> createVmList(final int vmCount, final String type) {
        List<Vm> vmList = new ArrayList<>(vmCount);

        for (int i = 0; i < vmCount; i++) {
            vmList.add(createVm(type));
            nextVmId++;
        }

        return vmList;
    }

    private Vm createVm(final String type) {
        int sizeMultiplier = getSizeMultiplier(type);

        Vm vm = new VmSimple(
            nextVmId,
            settings.getHostPeMips(),
            settings.getBasicVmPeCnt() * sizeMultiplier);
            vm
                .setRam(settings.getBasicVmRam() * sizeMultiplier)
                .setBw(settings.getBasicVmBw())
                .setSize(settings.getBasicVmSize())
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setDescription(type)
                .setShutDownDelay(settings.getVmShutdownDelay());
        
        return vm;
    }

    private List<Pe> createPeList() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < settings.getHostPeCnt(); i++) {
            peList.add(new PeSimple(settings.getHostPeMips(), new PeProvisionerSimple()));
        }

        return peList;
    }

    private void runForInternal(final double interval, final double target) {
        double adjustedInterval = interval;
        while (cloudSimPlus.runFor(adjustedInterval) < target) {
            adjustedInterval = Math.max(target - clock(), cloudSimPlus.getMinTimeBetweenEvents());
        }
    }

    public void runFor(final double interval) {
        if (!isRunning()) {
            throw new IllegalStateException("The simulation is not running - "
                + "please reset or create a new one!");
        }

        final long startTime = TimeMeasurement.startTiming();
        final double target = cloudSimPlus.clock() + interval;

        jobsFinishedWaitTimeLastInterval.clear();

        scheduleJobsUntil(target);
        runForInternal(interval, target);

        if (shouldPrintJobStats()) {
            printJobStats();
        }

        // the size of cloudletsCreatedList grows to huge numbers
        // as we re-schedule cloudlets when VMs get killed
        // to avoid OOMing we need to clear that list
        // it is a safe operation in our environment, because that list is only used in
        // CloudSimPlus when a VM is being upscaled (we don't do that)
        if (!settings.isStoreCreatedCloudletsDatacenterBroker()) {
            broker.getCloudletCreatedList().clear();
        }

        long elapsedTimeInNs = TimeMeasurement.calculateElapsedTime(startTime);
        String startTimeFormat = String.format("%.1f", clock() - interval);
        LOGGER.debug("runFor [" + startTimeFormat + " - " + clock() + "] took "
            + elapsedTimeInNs + "ns / " + elapsedTimeInNs / 1_000_000_000d + "s");
    }

    private boolean shouldPrintJobStats() {
        return (settings.isPrintJobsPeriodically() && (int) Math.round(clock()) % 1000 == 0)
            || !isRunning();
    }

    public void printJobStats() {
        LOGGER.info("All jobs: " + inputJobs.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : inputJobs) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for(Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info(e.getKey().toString() + ": " + e.getValue());
        }

        LOGGER.info("Jobs which are still queued");
        for(Cloudlet cloudlet : inputJobs) {
            if (Cloudlet.Status.QUEUED.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
        LOGGER.info("Jobs which are still executed");
        for(Cloudlet cloudlet : inputJobs) {
            if (Cloudlet.Status.INEXEC.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
    }

    private void printCloudlet(final Cloudlet cloudlet) {
        LOGGER.info("Cloudlet: " + cloudlet.getId());
        LOGGER.info("Number of PEs: " + cloudlet.getPesNumber());
        LOGGER.info("Total length in MIs: " + cloudlet.getTotalLength());
        LOGGER.info("Submission delay: " + originalSubmissionDelay.get(cloudlet.getId()));
        LOGGER.info("Started: " + cloudlet.getStartTime());
        final Vm vm = cloudlet.getVm();
        LOGGER.info("VM: " + vm.getId() + "(" + vm.getDescription() + ")"
            + " CPU: " + vm.getPesNumber() + "/" + vm.getMips() + " @ "
            + vm.getCpuPercentUtilization()
            + " RAM: " + vm.getRam().getAllocatedResource()
            + " Start time: " + vm.getStartTime()
            + " Stop time: " + vm.getFinishTime());
    }

    private boolean isAnyVmSuitableForCloudlet(Cloudlet cloudlet) {
        List<Vm> vmExecList = getBroker().getVmExecList();
        for (Vm vm : vmExecList) {
            if (vm.isSuitableForCloudlet(cloudlet)) {
                return true;
            }
        }
        return false;
    }

    private void addOnStartListener(Cloudlet cloudlet) {
        cloudlet.addOnStartListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug("Cloudlet:" + cloudlet.getId()
                    + " started running on vm "
                    + cloudlet.getVm().getId());
            }
        });
    }

    private void addOnFinishListener(Cloudlet cloudlet) {
        cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug("Cloudlet: " + cloudlet.getId()
                    + " that was running on vm "
                    + cloudlet.getVm().getId() + " finished.");
                jobsFinishedWaitTimeLastInterval.add(
                    cloudlet.getStartTime() - originalSubmissionDelay.get(cloudlet.getId()));
            }
        });
    }

    // TODO: I should consider submitting all jobs immediately when simulation starts
    // using the submission delay that I have on the dataset. I assume that it will work.
    // It will be practically the same and this code below will be eliminated, providing
    // a more smooth code flow.

    private void scheduleJobsUntil(final double target) {
        List<Cloudlet> jobsToSubmit = new ArrayList<>();
        Iterator<Cloudlet> it = unsubmittedJobs.iterator();
        previousLastSubmittedJobIndex = lastSubmittedJobIndex;

        LOGGER.debug(unsubmittedJobs.size() + " jobs waiting to be submitted");

        while (it.hasNext()) {
            Cloudlet cloudlet = it.next();
            if (cloudlet.getSubmissionDelay() > target) {
                continue;
            }
            // Do not schedule cloudlet if there are no suitable vms to run it
            if (!isAnyVmSuitableForCloudlet(cloudlet)) {
                unableToSubmitJobCount++;
                LOGGER.debug("no vm available for " + cloudlet.getId());
                continue;
            }
            // here we calculate how much time the job needs to be submitted
            cloudlet.setSubmissionDelay(Math.max(cloudlet.getSubmissionDelay() - clock(), 0));
            jobsToSubmit.add(cloudlet);
            it.remove();
            lastSubmittedJobIndex++;
        }

        if (!jobsToSubmit.isEmpty()) {
            submitCloudletsList(jobsToSubmit);
            // jobsToSubmit.clear();
        }
    }

    private void submitCloudletsList(final List<Cloudlet> jobsToSubmit) {
        LOGGER.debug("Submitting: " + jobsToSubmit.size() + " jobs");
        broker.submitCloudletList(jobsToSubmit);

        // we immediately clear up that list because it is not really
        // used anywhere but traversing it takes a lot of time
        // No need because the history of this list is already disabled by default
        // see: https://javadoc.io/doc/org.cloudsimplus/cloudsim-plus/latest/org/cloudsimplus/schedulers/cloudlet/CloudletSchedulerAbstract.html#enableCloudletSubmittedList()
        // broker.getCloudletSubmittedList().clear();
    }

    public boolean isRunning() {
        // if we don't have unfinished jobs, it doesn't make sense to execute any actions
        return cloudSimPlus.isRunning() && hasUnfinishedJobs();
    }

    public void terminate() {
        cloudSimPlus.terminate();
    }

    private boolean hasUnfinishedJobs() {
        return broker.getCloudletFinishedList().size() < inputJobs.size();
    }

    public int getLastCreatedVmId() {
        return nextVmId - 1;
    }

    public long getAllocatedCores() {
        final long reduce = broker
            .getVmExecList()
            .parallelStream()
            .map(Vm::getPesNumber)
            .reduce(Long::sum)
            .orElse(0L);
        return reduce;
    }

    public double[] getVmCpuUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] cpuPercentUsage = new double[input.size()];
    
        IntStream.range(0, input.size())
            .forEach(i -> cpuPercentUsage[i] = input.get(i).getCpuPercentUtilization());
        
        return cpuPercentUsage;
    }

    public int getSubmittedJobsCountLastInterval() {
        return lastSubmittedJobIndex - previousLastSubmittedJobIndex;
    }

    public int getSubmittedJobsCount() {
        return lastSubmittedJobIndex;
    }

    public  List<Double> getFinishedJobsWaitTimeLastInterval() {
        return jobsFinishedWaitTimeLastInterval;
    }
    
    public long getWaitingJobsCountLastInterval() {
        double start = clock() - settings.getTimestepInterval();
        return inputJobs.parallelStream()
            .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
            .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) >= start)
            .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
            .count();
    }

    public long getWaitingJobsCount() {
        return inputJobs.parallelStream()
            .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
            .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
            .count();
    }

    public double[] getVmMemoryUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] memPercentUsage = new double[input.size()];
    
        IntStream.range(0, input.size())
            .forEach(i -> memPercentUsage[i] = input.get(i).getRam().getPercentUtilization());
    
        return memPercentUsage;
    }

    public boolean addNewVm(final String type, final long hostId) {
        LOGGER.debug("Agent action: Create a " + type + " VM on host " + hostId);

        final Host host = datacenter.getHostById(hostId);
        
        if (host == Host.NULL) {
            LOGGER.debug("Vm creating ignored, no host with given id found");
            return false;
        }

        final Vm newVm = createVm(type);
        newVm.setDescription(type + "-" + hostId);

        if (!host.isSuitableForVm(newVm)) {
            LOGGER.debug("Vm creating ignored, host not suitable");
            // LOGGER.debug("Host MIPS:" + host.getVmScheduler().getTotalAvailableMips());
            LOGGER.debug("Host Vm List Size:" + host.getVmList().size());
            LOGGER.debug("Total Vm Exec List Size:" + broker.getVmExecList().size());
            return false;
        }

        vmCost.addNewVmToList(newVm);
        nextVmId++;

        // assuming average startup delay is 56s as in 10.48550/arXiv.2107.03467
        final double delay = settings.getVmStartupDelay();
        // TODO: instead of submissiondelay, maybe consider adding the vm boot up delay
        newVm.setSubmissionDelay(delay);
        broker.submitVm(newVm);
        LOGGER.debug("VM creating requested, delay: " + delay + " type: " + type);
        return true;
    }

    // if a vm is destroyed, this method returns true, otherwise false
    public boolean removeVm(final int index) {
        List<Vm> vmExecList = broker.getVmExecList();
        LOGGER.debug("vmExecList.size = " + vmExecList);

        if (vmExecList.isEmpty()) {
            LOGGER.warn("Can't kill VM. No VMs running.");
            return false;
        }

        Vm vmToKill = vmExecList.get(index);

        destroyVm(vmToKill);
        return true;
    }

    private Cloudlet resetCloudlet(final Cloudlet cloudlet) {
        return cloudlet.setVm(Vm.NULL).reset();
    }

    private List<Cloudlet> resetCloudlets(List<CloudletExecution> cloudlets) {
        return cloudlets
            .parallelStream()
            .map(CloudletExecution::getCloudlet)
            .map(this::resetCloudlet)
            .collect(Collectors.toList());
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

        datacenter.getVmAllocationPolicy().deallocateHostForVm(vm); // this internally calls the destroyVm

        vm.getCloudletScheduler().clear();

        LOGGER.debug("Killing VM: " + vm.getId() + ", to-reschedule cloudlets count: "
            + affectedCloudlets.size() + ", type: " + vmSize);
        if (!affectedCloudlets.isEmpty()) {
            rescheduleCloudlets(affectedCloudlets);
        }
    }

    private void rescheduleCloudlets(final List<Cloudlet> affectedCloudlets) {
        final double currentClock = clock();

        affectedCloudlets.forEach(cloudlet -> {
            Double submissionDelay = originalSubmissionDelay.get(cloudlet.getId());
            // if the Cloudlet still hasn't been started, 
            // let it start at the scheduled time,
            // else, start it immediately
            // we can also start it on the next timestepInterval
            // THIS IS NOT NEEDED ACTUALLY. There is no way the cloudlet
            // is not already submitted because it was in the vm's queue.
            submissionDelay = Math.max(0, submissionDelay - currentClock);
            cloudlet.setSubmissionDelay(submissionDelay);
        });

        unsubmittedJobs.addAll(affectedCloudlets);
    }

    public CloudSimPlus getSimulation() {
        return cloudSimPlus;
    }

    public DatacenterBrokerFirstFitFixed getBroker() {
        return broker;
    }

    public Datacenter getDatacenter() {
        return datacenter;
    }

    public double clock() {
        return cloudSimPlus.clock();
    }

    public long getNumberOfFutureEvents() {
        return cloudSimPlus.getNumberOfFutureEvents(simEvent -> true);
    }

    public double getRunningCost() {
        return vmCost.getVMCostPerIteration(clock());
    }
    
    public int getUnableToSubmitJobCount() {
        return unableToSubmitJobCount;
    }
}

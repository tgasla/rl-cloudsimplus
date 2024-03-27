package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.HostAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.listeners.EventListener;

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
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CloudSimProxy {

    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};

    private static final Logger LOGGER = 
            LoggerFactory.getLogger(CloudSimProxy.class.getSimpleName());

    private final DatacenterBrokerFirstFitFixed broker;
    private final CloudSimPlus cloudSimPlus;
    private final SimulationSettings settings;
    private final VmCost vmCost;
    private final Datacenter datacenter;
    private final double simulationSpeedUp;
    private final Map<Long, Double> originalSubmissionDelay = new HashMap<>();
    private final Random random = new Random(System.currentTimeMillis());
    private final List<Cloudlet> jobs = new ArrayList<>();
    private final List<Cloudlet> potentiallyWaitingJobs = new ArrayList<>(1024);
    private final List<Cloudlet> alreadyStarted = new ArrayList<>(128);
    private final Set<Long> finishedIds = new HashSet<>();
    private int toAddJobId = 0;
    private int previousIntervalJobId = 0;
    private int nextVmId;

    public CloudSimProxy(final SimulationSettings settings,
                         final Map<String, Integer> initialVmsCount,
                         final List<Cloudlet> inputJobs,
                         final double simulationSpeedUp) {
        this.settings = settings;
        this.cloudSimPlus = new CloudSimPlus(0.01);
        this.broker = new DatacenterBrokerFirstFitFixed(cloudSimPlus);
        this.datacenter = createDatacenter();
        this.vmCost = new VmCost(
                settings.getVmRunningHourlyCost(),
                simulationSpeedUp,
                settings.isPayingForTheFullHour());
        this.simulationSpeedUp = simulationSpeedUp;

        this.nextVmId = 0;

        final List<? extends Vm> smallVmList = createVmList(initialVmsCount.get(SMALL), SMALL);
        final List<? extends Vm> mediumVmList = createVmList(initialVmsCount.get(MEDIUM), MEDIUM);
        final List<? extends Vm> largeVmList = createVmList(initialVmsCount.get(LARGE), LARGE);
        broker.submitVmList(smallVmList);
        broker.submitVmList(mediumVmList);
        broker.submitVmList(largeVmList);

        this.jobs.addAll(inputJobs);
        Collections.sort(this.jobs, new DelayCloudletComparator());
        this.jobs.forEach(c -> originalSubmissionDelay.put(c.getId(), c.getSubmissionDelay()));

        scheduleAdditionalCloudletProcessingEvent(this.jobs);

        this.cloudSimPlus.startSync();
        this.runFor(0.1);
    }

    public boolean allJobsFinished() {
        return this.finishedIds.size() == this.jobs.size();
    }

    public int getFinishedCount() {
        return finishedIds.size();
    }

    private void scheduleAdditionalCloudletProcessingEvent(final List<Cloudlet> jobs) {
        // a second after every cloudlet will be submitted we add an event - this should prevent
        // the simulation from ending while we have some jobs to schedule
        jobs.forEach(c ->
                this.cloudSimPlus.send(
                        datacenter,
                        datacenter,
                        c.getSubmissionDelay() + 1.0,
                        CloudSimTag.VM_UPDATE_CLOUDLET_PROCESSING,
                        null
                )
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

        return new LoggingDatacenter(cloudSimPlus, hostList, new VmAllocationPolicySimple());
    }

    private List<? extends Vm> createVmList(final int vmCount, final String type) {
        List<Vm> vmList = new ArrayList<>(vmCount);

        for (int i = 0; i < vmCount; i++) {
            // 1 VM == 1 HOST for simplicity
            vmList.add(createVm(type));
        }

        return vmList;
    }

    private Vm createVm(final String type) {
        int sizeMultiplier = getSizeMultiplier(type);

        Vm vm = new VmSimple(
                this.nextVmId,
                settings.getHostPeMips(),
                settings.getBasicVmPeCnt() * sizeMultiplier);
                vm
                .setRam(settings.getBasicVmRam() * sizeMultiplier)
                .setBw(settings.getBasicVmBw())
                .setSize(settings.getBasicVmSize())
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setDescription(type)
                .setShutDownDelay(settings.getVmShutdownDelay());

        vmCost.addNewVmToList(vm);
        this.nextVmId++;
        
        return vm;
    }

    public int getSizeMultiplier(final String type) {
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

    private List<Pe> createPeList() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < settings.getHostPeCnt(); i++) {
            peList.add(new PeSimple(settings.getHostPeMips(), new PeProvisionerSimple()));
        }

        return peList;
    }

    public void runFor(final double interval) {
        if (!this.isRunning()) {
            throw new RuntimeException("The simulation is not running - "
                    + "please reset or create a new one!");
        }

        long start = System.nanoTime();
        final double target = this.cloudSimPlus.clock() + interval;

        scheduleJobsUntil(target);

        int i = 0;
        double adjustedInterval = interval;
        while (this.cloudSimPlus.runFor(adjustedInterval) < target) {
            adjustedInterval = target - this.cloudSimPlus.clock();
            adjustedInterval = adjustedInterval <= 0 
                    ? cloudSimPlus.getMinTimeBetweenEvents() : adjustedInterval;

            // Force stop if something runs out of control
            if (i >= 10000) {
                throw new RuntimeException("Breaking a really long loop in runFor!");
            }
            i++;
        }

        alreadyStarted.clear();

        final Iterator<Cloudlet> iterator = potentiallyWaitingJobs.iterator();
        while (iterator.hasNext()) {
            Cloudlet job = iterator.next();
            if (job.getStatus() == Cloudlet.Status.INEXEC 
                    || job.getStatus() == Cloudlet.Status.SUCCESS
                    || job.getStatus() == Cloudlet.Status.CANCELED) {

                alreadyStarted.add(job);
                iterator.remove();
            }
        }

        cancelInvalidEvents();
        printJobStatsAfterEndOfSimulation();

        if (shouldPrintJobStats()) {
            printJobStats();
        }

        // the size of cloudletsCreatedList grows to huge numbers
        // as we re-schedule cloudlets when VMs get killed
        // to avoid OOMing we need to clear that list
        // it is a safe operation in our environment, because that list is only used in
        // CloudSim+ when a VM is being upscaled (we don't do that)
        if (!settings.isStoreCreatedCloudletsDatacenterBroker()) {
            this.broker.getCloudletCreatedList().clear();
        }

        long end = System.nanoTime();
        long diff = end - start;
        double diffInSec = ((double) diff) / 1_000_000_000L;

        // TODO: can be removed after validating the fix of OOM
        // should always be zero
        final int debugBrokerCreatedListSize = this.broker.getCloudletCreatedList().size();
        LOGGER.debug("runFor (" + this.clock() + ") took " 
                + diff + "ns / " + diffInSec + "s (DEBUG: " + debugBrokerCreatedListSize + ")");
    }

    private boolean shouldPrintJobStats() {
        return settings.getPrintJobsPeriodically() 
                && Double.valueOf(this.clock()).longValue() % 20000 == 0;
    }

    private void printJobStatsAfterEndOfSimulation() {
        if (!this.isRunning()) {
            // LOGGER.info("CloudSimProxy.isRunning: " + cloudSimPlus.isRunning());
            LOGGER.info("End of simulation, some reality check stats:");
            printJobStats();
        }
    }

    public void printJobStats() {
        LOGGER.info("All jobs: " + this.jobs.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : this.jobs) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for(Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info(e.getKey().toString() + ": " + e.getValue());
        }

        LOGGER.info("Jobs which are still queued");
        for(Cloudlet cloudlet : this.jobs) {
            if (Cloudlet.Status.QUEUED.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
        LOGGER.info("Jobs which are still executed");
        for(Cloudlet cloudlet : this.jobs) {
            if (Cloudlet.Status.INEXEC.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
    }

    private void printCloudlet(final Cloudlet cloudlet) {
        LOGGER.info("Cloudlet: " + cloudlet.getId());
        LOGGER.info("Number of PEs: " + cloudlet.getPesNumber());
        LOGGER.info("Number of MIPS: " + cloudlet.getLength());
        LOGGER.info("Submission delay: " + cloudlet.getSubmissionDelay());
        LOGGER.info("Started: " + cloudlet.getStartTime());
        final Vm vm = cloudlet.getVm();
        LOGGER.info("VM: " + vm.getId() + "(" + vm.getDescription() + ")"
                + " CPU: " + vm.getPesNumber() + "/" + vm.getMips() + " @ "
                + vm.getCpuPercentUtilization()
                + " RAM: " + vm.getRam().getAllocatedResource()
                + " Start time: " + vm.getStartTime()
                + " Stop time: " + vm.getFinishTime());
    }

    private void cancelInvalidEvents() {
        final long clock = (long) cloudSimPlus.clock();

        if (clock % 100 == 0) {
            LOGGER.debug("Cleaning up events (before): " + getNumberOfFutureEvents());
            cloudSimPlus.cancelAll(datacenter, new Predicate<SimEvent>() {

                private SimEvent previous;

                @Override
                public boolean test(SimEvent current) {
                    // remove dupes
                    if (previous != null
                            && current.getTag() == CloudSimTag.VM_UPDATE_CLOUDLET_PROCESSING
                            && current.getSource() == datacenter
                            && current.getDestination() == datacenter
                            && previous.getTime() == current.getTime()
                            && current.getData() == null
                    ) {
                        return true;
                    }

                    previous = current;
                    return false;
                }
            });
            LOGGER.debug("Cleaning up events (after): " + getNumberOfFutureEvents());
        }
    }

    private void scheduleJobsUntil(final double target) {
        previousIntervalJobId = nextVmId;
        List<Cloudlet> jobsToSubmit = new ArrayList<>();

        while (toAddJobId < this.jobs.size() && 
                this.jobs.get(toAddJobId).getSubmissionDelay() <= target) {
            // we process every cloudlet only once here...
            final Cloudlet cloudlet = this.jobs.get(toAddJobId);

            // the job should enter the cluster once target is crossed
            cloudlet.setSubmissionDelay(1.0);
            cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
                @Override
                public void update(CloudletVmEventInfo info) {
                    LOGGER.debug("Cloudlet: " + cloudlet.getId() + "/" + 
                            cloudlet.getVm().getId() + " Finished.");
                    finishedIds.add(info.getCloudlet().getId());
                }
            });
            jobsToSubmit.add(cloudlet);
            toAddJobId++;
        }

        if (jobsToSubmit.size() > 0) {
            submitCloudletsList(jobsToSubmit);
            potentiallyWaitingJobs.addAll(jobsToSubmit);
        }

        int pctSubmitted = (int) ((toAddJobId / (double) this.jobs.size()) * 10000d);
        int upper = pctSubmitted / 100;
        int lower = pctSubmitted % 100;
        LOGGER.info(
                "Simulation progress: submitted: " + upper + "." + lower + "% "
                + "Waiting list size: " + this.broker.getCloudletWaitingList().size());
    }

    private void submitCloudletsList(final List<Cloudlet> jobsToSubmit) {
        LOGGER.debug("Submitting: " + jobsToSubmit.size() + " jobs");
        broker.submitCloudletList(jobsToSubmit);

        // we immediately clear up that list because it is not really
        // used anywhere but traversing it takes a lot of time
        broker.getCloudletSubmittedList().clear();
    }

    public boolean isRunning() {
        // if we don't have unfinished jobs, it doesn't make sense to execute
        // any actions
        return cloudSimPlus.isRunning() && hasUnfinishedJobs();
    }

    private boolean hasUnfinishedJobs() {
        return this.finishedIds.size() < this.jobs.size();
    }

    public int getLastCreatedVmId() {
        return nextVmId - 1;
    }

    public long getNumberOfActiveCores() {
        final Optional<Long> reduce = this.broker
                .getVmExecList()
                .parallelStream()
                .map(Vm::getPesNumber)
                .reduce(Long::sum);
        return reduce.orElse(0L);
    }

    public double[] getVmCpuUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] cpuPercentUsage = new double[input.size()];
        int i = 0;
        for (Vm vm : input) {
            cpuPercentUsage[i] = vm.getCpuPercentUtilization();
            i++;
        }

        return cpuPercentUsage;
    }

    public int getSubmittedJobsCountLastInterval() {
        return toAddJobId - previousIntervalJobId;
    }

    public int getWaitingJobsCountInterval(final double interval) {
        double start = clock() - interval;

        int jobsWaitingSubmittedInTheInterval = 0;
        for (Cloudlet cloudlet : potentiallyWaitingJobs) {
            if (!cloudlet.getStatus().equals(Cloudlet.Status.INEXEC)) {
                double systemEntryTime = this.originalSubmissionDelay.get(cloudlet.getId());
                if (systemEntryTime >= start) {
                    jobsWaitingSubmittedInTheInterval++;
                }
            }
        }
        return jobsWaitingSubmittedInTheInterval;
    }

    public int getSubmittedJobsCount() {
        // this is incremented every time job is submitted
        return this.toAddJobId;
    }

    public double[] getVmMemoryUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] memPercentUsage = new double[input.size()];
        int i = 0;
        for (Vm vm : input) {
            memPercentUsage[i] = vm.getRam().getPercentUtilization();
        }
        return memPercentUsage;
    }

    public boolean addNewVm(final String type, final long vmId) {
        LOGGER.debug("Agent action: Create a " + type + " VM");

        final long hostId = broker.getVmExecList()
                .parallelStream()
                .filter(vm -> vmId == vm.getId())
                .findFirst()
                .map(Vm::getHost)
                .map(Host::getId)
                .orElse(-1L);
        
        if (hostId == -1L) {
            LOGGER.debug("Vm creating ignored, no vm with id given found");
            return false;
        }

        Vm newVm = createVm(type);
        newVm.setDescription(type + "-" + hostId);

        // assuming average delay up to 97s as in 10.1109/CLOUD.2012.103
        // from anecdotal exp the startup time can be as fast as 45s
        final double delay = (45 + Math.random() * 52) / this.simulationSpeedUp;
        // TODO: instead of submissiondelay, maybe consider adding the vm boot up delay
        newVm.setSubmissionDelay(delay);
        broker.submitVm(newVm);
        LOGGER.debug("VM creating requested, delay: " + delay + " type: " + type);
        return true;
    }

    public void addNewVm(String type) {
        Vm newVm = createVm(type);
        // assuming average delay up to 97s as in 10.1109/CLOUD.2012.103
        // from anecdotal exp the startup time can be as fast as 45s
        final double delay = (45 + Math.random() * 52) / this.simulationSpeedUp;
        // TODO: instead of submissiondelay, maybe consider adding the vm boot up delay
        newVm.setSubmissionDelay(delay);
        LOGGER.debug("Agent action: Create a " + type + " VM");
        broker.submitVm(newVm);
        LOGGER.debug("VM creating requested, delay: " + delay + " type: " + type);
    }

    // if a vm is destroyed, this method returns the type of it.
    public String removeVm(final long id) {
        final Vm vmToKill = broker.getVmExecList()
                .parallelStream()
                .filter(vm -> id == vm.getId())
                .findFirst()
                .orElse(Vm.NULL);

        if (vmToKill == Vm.NULL) {
            LOGGER.warn("Can't kill the VM with id " + id + ". No such vm found.");
            return null;
        }

        String vmType = vmToKill.getDescription();

        LOGGER.debug("vmType = " + vmType);

        List<Vm> vmsOfType = broker.getVmExecList()
                .parallelStream()
                .filter(vm -> vmType.equals((vm.getDescription())))
                .collect(Collectors.toList());
        LOGGER.debug("Agent action: Remove VM with id " + id);
        if (canKillVm(vmType, vmsOfType.size())) {
            destroyVm(vmToKill);
            return vmType;
        } else {
            LOGGER.warn("Can't kill the VM. It is the only SMALL one running");
            return null;
        }
    }

    public boolean removeRandomVm(String type) {
        List<Vm> vmExecList = broker.getVmExecList();

        List<Vm> vmsOfType = vmExecList
                .parallelStream()
                .filter(vm -> type.equals((vm.getDescription())))
                .collect(Collectors.toList());
        LOGGER.debug("Agent action: Remove a " + type + " VM");
        if (canKillVm(type, vmsOfType.size())) {
            int vmToKillIdx = random.nextInt(vmsOfType.size());
            destroyVm(vmsOfType.get(vmToKillIdx));
            return true;
        } else {
            LOGGER.warn("Can't kill the VM. It is the only SMALL one running");
            return false;
        }
    }

    private boolean canKillVm(final String type, final int size) {
        if (SMALL.equals(type)) {
            return size > 1;
        }

        return size > 0;
    }

    private Cloudlet resetCloudlet(final Cloudlet cloudlet) {
        cloudlet.setVm(Vm.NULL);
        return cloudlet.reset();
    }

    private List<Cloudlet> resetCloudlets(List<CloudletExecution> cloudlets) {
        return cloudlets
                .stream()
                .map(CloudletExecution::getCloudlet)
                .map(this::resetCloudlet)
                .collect(Collectors.toList());
    }

    private void destroyVm(Vm vm) {
        final String vmSize = vm.getDescription();

        /* TODO: it turned out that when we clean the submitted list, 
         * we probably "forget" to execute some cloudlets
         * it seems to me that there is simply some list in the scheduler that we forget about
         * and we need to take into account
         * 
         * I think this is now fixed, have to double-check it though.
         */

        // replaces broker.destroyVm
        final List<Cloudlet> affectedExecCloudlets = 
                resetCloudlets(vm.getCloudletScheduler().getCloudletExecList());
        final List<Cloudlet> affectedWaitingCloudlets =
                resetCloudlets(vm.getCloudletScheduler().getCloudletWaitingList());
        final List<Cloudlet> affectedCloudlets =
                Stream.concat(affectedExecCloudlets.stream(),affectedWaitingCloudlets.stream())
                .collect(Collectors.toList());
        ((HostAbstract)vm.getHost()).destroyVm(vm);
        vm.getCloudletScheduler().clear();
        // replaces broker.destroyVm

        LOGGER.debug("Killing VM: "
                + vm.getId()
                + ", to-reschedule cloudlets count: "
                + affectedCloudlets.size()
                + ", type: "
                + vmSize);
        if (affectedCloudlets.size() > 0) {
            rescheduleCloudlets(affectedCloudlets);
        }
    }

    private void rescheduleCloudlets(final List<Cloudlet> affectedCloudlets) {
        final double currentClock = cloudSimPlus.clock();

        affectedCloudlets.forEach(cloudlet -> {
            Double submissionDelay = originalSubmissionDelay.get(cloudlet.getId());

            if (submissionDelay == null) {
                throw new RuntimeException("Cloudlet with ID: " 
                        + cloudlet.getId() 
                        + " not seen previously! Original submission time unknown!");
            }

            if (submissionDelay < currentClock) {
                submissionDelay = 1.0;
            } else {
                // if the Cloudlet still hasn't been started, 
                // let it start at the scheduled time.
                submissionDelay -= currentClock;
            }

            cloudlet.setSubmissionDelay(submissionDelay);
        });

        long brokerStart = System.nanoTime();
        submitCloudletsList(affectedCloudlets);
        long brokerStop = System.nanoTime();

        double brokerTime = (brokerStop - brokerStart) / 1_000_000_000d;
        LOGGER.debug("Rescheduling "
                + affectedCloudlets.size()
                + " cloudlets took (s): "
                + brokerTime);
    }

    public double clock() {
        return this.cloudSimPlus.clock();
    }

    public long getNumberOfFutureEvents() {
        return this.cloudSimPlus.getNumberOfFutureEvents(simEvent -> true);
    }

    public int getWaitingJobsCount() {
        return this.potentiallyWaitingJobs.size();
    }

    public double getRunningCost() {
        return vmCost.getVMCostPerIteration(this.clock());
    }

    class DelayCloudletComparator implements Comparator<Cloudlet> {

        @Override
        public int compare(Cloudlet left, Cloudlet right) {
            final double diff = left.getSubmissionDelay() - right.getSubmissionDelay();
            if (diff < 0) {
                return -1;
            }

            if (diff > 0) {
                return 1;
            }
            return 0;
        }
    }
}

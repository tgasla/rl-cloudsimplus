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
import java.util.stream.IntStream;

public class CloudSimProxy {

    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = {SMALL, MEDIUM, LARGE};

    private static final Logger LOGGER = 
        LoggerFactory.getLogger(CloudSimProxy.class.getSimpleName());
    private static final double minTimeBetweenEvents = 0.1;

    private final DatacenterBrokerFirstFitFixed broker;
    private final CloudSimPlus cloudSimPlus;
    private final SimulationSettings settings;
    private final VmCost vmCost;
    private final Datacenter datacenter;
    private final Map<Long, Double> originalSubmissionDelay = new HashMap<>();
    private final Random random = new Random(System.currentTimeMillis());
    private final List<Cloudlet> jobs = new ArrayList<>();
    private final List<Cloudlet> submittedJobs = new ArrayList<>(1024);
    private final List<Cloudlet> jobsFinishedThisTimestep = new ArrayList<>(128);
    private CsvWriter jobCsvWriter;
    // private final CsvWriter hostCsvWriter;
    private int nextVmId = 0;
    private int lastSubmittedJobIndex = 0;
    private int previousTimestepLastSubmittedJobIndex = 0;
    private int episodeCount;

    public CloudSimProxy(
        final SimulationSettings settings,
        final List<Cloudlet> inputJobs,
        final int episodeCount
    ) {
        this.settings = settings;
        this.episodeCount = episodeCount;
        jobCsvWriter = null;

        String jobLogDir = settings.getJobLogDir();
        // String hostLogDir = jobLogDir;
        
        String[] jobCsvHeader = {
            // "jobId", 
            // "hostId",
            // "vmId",
            // "vmType",
            // "arrivalTime",
            "getStartWaitTime"
            // "execFinishTime"
        };

        // if (episodeCount == 1) {
        // jobCsvWriter = new CsvWriter(jobLogDir, "job_log.csv", jobCsvHeader);
        // }

        // String[] hostCsvHeader = {"hostId", "vmsRunningCount", "coresUtilized"};
        jobCsvWriter = new CsvWriter(jobLogDir, "job_log.csv", jobCsvHeader);
        // hostCsvWriter = new CsvWriter(hostLogDir, "host_log.csv", hostCsvHeader);
        
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

        jobs.addAll(inputJobs);
        Collections.sort(jobs, new DelayCloudletComparator());
        jobs.forEach(c -> originalSubmissionDelay.put(c.getId(), c.getSubmissionDelay()));

        cloudSimPlus.addOnEventProcessingListener(new EventListener<SimEvent>() {
            @Override
            public void update(SimEvent info) {
                if (getNumberOfFutureEvents() == 1 && hasUnfinishedJobs()) {
                    LOGGER.debug("There are unfinished jobs. "
                        + "Sending a NONE event to prevent simulation from ending.");
                    // this will prevent the simulation from ending
                    // while some jobs have not yet finished running
                    sendEmptyEventAt(settings.getTimestepInterval() + 1);
                }
            }
        });

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

    private void sendEmptyEventAt(double time) {
        // We add empty events to prevent the simulation from ending
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

    private void removeAlreadyStartedJobsFromList(List<Cloudlet> submittedJobs) {
        final Iterator<Cloudlet> iterator = submittedJobs.iterator();
        while (iterator.hasNext()) {
            Cloudlet job = iterator.next();
            if (job.getStatus() == Cloudlet.Status.INEXEC 
                || job.getStatus() == Cloudlet.Status.SUCCESS
                || job.getStatus() == Cloudlet.Status.CANCELED) {
                iterator.remove();
            }
        }
    }

    private void runForInternal(double interval, final double target) {
        while (cloudSimPlus.runFor(interval) < target) {
            interval = Math.max(target - clock(), cloudSimPlus.getMinTimeBetweenEvents());
        }
    }

    public void runFor(final double interval) {
        if (!isRunning()) {
            throw new RuntimeException("The simulation is not running - "
                + "please reset or create a new one!");
        }

        long startTime = TimeMeasurement.startTiming();

        final double target = cloudSimPlus.clock() + interval;

        scheduleJobsUntil(target);
        runForInternal(interval, target);

        removeAlreadyStartedJobsFromList(submittedJobs);

        printJobStatsAfterEndOfSimulation();
        closeCsvAfterEndOfSimulation();

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

        // List<Host> hostList = datacenter.getHostList();
        // int[] vmsRunningCount = new int[hostList.size()]; 
        // int[] pesUtilized = new int[hostList.size()];
        // int i = 0;
        // for (Host host : hostList) {
        //     vmsRunningCount[i] = host.getVmList().size();
        //     pesUtilized[i] = host.getBusyPesNumber();
        //     i++;
        // }

        // Object[] csvRow = {
        //     vmsRunningCount,
        //     pesUtilized
        // };

        // hostCsvWriter.writeRow(csvRow);

        long elapsedTimeInNs = TimeMeasurement.calculateElapsedTime(startTime);
        LOGGER.debug("runFor (" + clock() + ") took "
            + elapsedTimeInNs + "ns / " + elapsedTimeInNs / 1_000_000_000d + "s");
    }

    private boolean shouldPrintJobStats() {
        return settings.getPrintJobsPeriodically() 
            && Double.valueOf(clock()).longValue() % 20000 == 0;
    }

    private void printJobStatsAfterEndOfSimulation() {
        if (!isRunning()) {
            // LOGGER.info("cloudSimPlus.isRunning: " + cloudSimPlus.isRunning());
            LOGGER.info("End of simulation, some reality check stats:");
            printJobStats();
        }
    }

    private void closeCsvAfterEndOfSimulation() {
        if (!isRunning() && jobCsvWriter != null) {
            jobCsvWriter.close();
        }
        // if (!isRunning() && hostCsvWriter != null) {
        //     hostCsvWriter.close();
        // }
    }

    public void printJobStats() {
        LOGGER.info("All jobs: " + jobs.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : jobs) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for(Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info(e.getKey().toString() + ": " + e.getValue());
        }

        LOGGER.info("Jobs which are still queued");
        for(Cloudlet cloudlet : jobs) {
            if (Cloudlet.Status.QUEUED.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
        LOGGER.info("Jobs which are still executed");
        for(Cloudlet cloudlet : jobs) {
            if (Cloudlet.Status.INEXEC.equals(cloudlet.getStatus())) {
                printCloudlet(cloudlet);
            }
        }
    }

    private void printCloudlet(final Cloudlet cloudlet) {
        LOGGER.info("Cloudlet: " + cloudlet.getId());
        LOGGER.info("Number of PEs: " + cloudlet.getPesNumber());
        LOGGER.info("Total length in MIs: " + cloudlet.getTotalLength());
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

    private boolean isAnyVmSuitableForCloudlet(Cloudlet cloudlet) {
        List<Vm> vmExecList = getBroker().getVmExecList();
        for (Vm vm : vmExecList) {
            if (vm.isSuitableForCloudlet(cloudlet)) {
                return true;
            }
        }
        return false;
    }

    // TODO: I should consider submitting all jobs immediately when simulation starts
    // using the submission delay that I have on the dataset. I assume that it will work.
    // It will be practically the same and this code below will be eliminated, providing
    // a more smooth code flow.
    private void scheduleJobsUntil(final double target) {
        previousTimestepLastSubmittedJobIndex = lastSubmittedJobIndex;

        List<Cloudlet> jobsToSubmit = new ArrayList<>();

        jobsFinishedThisTimestep.clear();
        
        for (int jobIndexToTry = lastSubmittedJobIndex; jobIndexToTry < jobs.size(); jobIndexToTry++) {
            final Cloudlet cloudlet = jobs.get(jobIndexToTry);
            if (cloudlet.getSubmissionDelay() > target) {
                continue;
            }
            // Do not schedule cloudlet if there are no suitable vms to run it
            if (!isAnyVmSuitableForCloudlet(cloudlet)) {
                continue;
            }
            // final double cloudletArrivalTime = cloudlet.getSubmissionDelay();

            // the job should enter the cluster once target is crossed
            cloudlet.setSubmissionDelay(settings.getTimestepInterval());
            jobsToSubmit.add(cloudlet);
            lastSubmittedJobIndex++;

            cloudlet.addOnStartListener(new EventListener<CloudletVmEventInfo>() {
                @Override
                public void update(CloudletVmEventInfo info) {
                    LOGGER.debug("Cloudlet:" + cloudlet.getId()
                        + " started running on vm "
                        + cloudlet.getVm().getId());
                }
            });

            cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
                @Override
                public void update(CloudletVmEventInfo info) {
                    LOGGER.debug("Cloudlet: " + cloudlet.getId()
                        + " that was running on vm "
                        + cloudlet.getVm().getId() + " finished.");
                    
                    if (episodeCount == 1) {
                        Object[] csvRow = {
                            // cloudlet.getId(),
                            // cloudlet.getVm().getHost(),
                            // cloudlet.getVm().getId(),
                            // cloudlet.getVm().getDescription(),
                            // originalSubmissionDelay.get(cloudlet.getId()) // cloudlet.getDcArrivalTime(),
                            cloudlet.getStartTime() - originalSubmissionDelay.get(cloudlet.getId())// cloudlet.getStartWaitTime()
                            // cloudSimPlus.clock()
                        };
                        jobCsvWriter.writeRow(csvRow);
                    }
                    // finishedIds.add(info.getCloudlet().getId());
                    jobsFinishedThisTimestep.add(info.getCloudlet());
                }
            });
        }

        if (!jobsToSubmit.isEmpty()) {
            submitCloudletsList(jobsToSubmit);
            submittedJobs.addAll(jobsToSubmit);
            int pctSubmitted = (int) ((lastSubmittedJobIndex / (double) jobs.size()) * 10000d);
            int upper = pctSubmitted / 100;
            int lower = pctSubmitted % 100;
            LOGGER.info(
                "Simulation progress: submitted: " + upper + "." + lower + "% "
                + "Waiting list size: " + broker.getCloudletWaitingList().size());
        }
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
        return broker.getCloudletFinishedList().size() < jobs.size();
    }

    public int getLastCreatedVmId() {
        return nextVmId - 1;
    }

    public long getNumberOfActiveCores() {
        final Optional<Long> reduce = broker
            .getVmExecList()
            .parallelStream()
            .map(Vm::getPesNumber)
            .reduce(Long::sum);
        return reduce.orElse(0L);
    }

    public double[] getVmCpuUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] cpuPercentUsage = new double[input.size()];
    
        IntStream.range(0, input.size())
            .forEach(i -> cpuPercentUsage[i] = input.get(i).getCpuPercentUtilization());
    
        return cpuPercentUsage;
    }

    public int getSubmittedJobsCountLastInterval() {
        return lastSubmittedJobIndex - previousTimestepLastSubmittedJobIndex;
    }
    
    public long getWaitingJobsCountLastInterval() {
        double start = clock() - settings.getTimestepInterval();
        return submittedJobs.parallelStream()
            .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
            .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) >= start)
            .count();
    }
    

    public int getSubmittedJobsCount() {
        // this is incremented every time job is submitted
        return lastSubmittedJobIndex;
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

    // TODO: I should avoid repeating code here.
    // I should call addNewVm(type, vmId) inside this method
    // public void addNewVm(String type) {
    //     Vm newVm = createVm(type);
    //     // assuming average delay up to 97s as in 10.1109/CLOUD.2012.103
    //     // from anecdotal exp the startup time can be as fast as 45s
    //     final double delay = 
    //             (45 + Math.random() * 52) / settings.getSimulationSpeedup();
    //     // TODO: instead of submissiondelay, maybe consider adding the vm boot up delay
    //     // newVm.setSubmissionDelay(delay);
    //     LOGGER.debug("Agent action: Create a " + type + " VM");
    //     broker.submitVm(newVm);
    //     LOGGER.debug("VM creating requested, delay: " + delay + " type: " + type);
    // }

    // if a vm is destroyed, this method returns the type of it.
    public boolean removeVm(final int index) {
        List<Vm> vmExecList = broker.getVmExecList();
        LOGGER.debug("vmExecList.size = " + vmExecList);

        if (vmExecList.isEmpty()) {
            LOGGER.warn("Can't kill VM. No VMs running.");
            return false;
        }

        Vm vmToKill = vmExecList.get(index);

        // if (vmExecList.size() == 1) {
        //     LOGGER.warn("Can't kill VM as it is the only one running.");
        //     return false;
        // }

        destroyVm(vmToKill);
        return true;
    }

    // TODO: I should avoid repeating code here.
    // I should reuse code from removeVm(id)
    // public boolean removeRandomVm(String type) {
    //     List<Vm> vmExecList = broker.getVmExecList();

    //     if (vmExecList.size() == 1) {
    //         LOGGER.warn("Can't kill VM as it is the only one running.");
    //         return false;
    //     }

    //     List<Vm> vmsOfType = vmExecList
    //         .parallelStream()
    //         .filter(vm -> type.equals((vm.getDescription())))
    //         .collect(Collectors.toList());
    //     LOGGER.debug("Agent action: Remove a " + type + " VM");

    //     if (vmsOfType.size() == 0) {
    //         LOGGER.warn("Can't kill VM of type " + type + " as no vms of this type are running");
    //         return false;
    //     }

    //     int vmToKillIdx = random.nextInt(vmsOfType.size());
    //     destroyVm(vmsOfType.get(vmToKillIdx));
    //     return true;
    // }

    private Cloudlet resetCloudlet(final Cloudlet cloudlet) {
        cloudlet.setVm(Vm.NULL);
        return cloudlet.reset();
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

        final List<Cloudlet> affectedExecCloudlets = 
            resetCloudlets(vm.getCloudletScheduler().getCloudletExecList());
        final List<Cloudlet> affectedWaitingCloudlets =
            resetCloudlets(vm.getCloudletScheduler().getCloudletWaitingList());
        final List<Cloudlet> affectedCloudlets =
            Stream.concat(affectedExecCloudlets.stream(),affectedWaitingCloudlets.stream())
                .collect(Collectors.toList());
        ((HostAbstract)vm.getHost()).destroyVm(vm);
        
        vm.getCloudletScheduler().clear();

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
        final double currentClock = clock();
        final double timestepInterval = settings.getTimestepInterval();

        affectedCloudlets.forEach(cloudlet -> {
            Double submissionDelay = originalSubmissionDelay.get(cloudlet.getId());

            if (submissionDelay == null) {
                throw new RuntimeException("Cloudlet with ID: " 
                + cloudlet.getId() 
                + " not seen previously! Original submission time unknown!");
            }

            // if the Cloudlet still hasn't been started, 
            // let it start at the scheduled time,
            // else, start it immediately
            submissionDelay = Math.max(0, submissionDelay - currentClock);
            cloudlet.setSubmissionDelay(submissionDelay);
        });

        long startTime = TimeMeasurement.startTiming();
        submitCloudletsList(affectedCloudlets);
        long elapsedTimeInNs = TimeMeasurement.calculateElapsedTime(startTime);

        double brokerTime = elapsedTimeInNs / 1_000_000_000d;
        LOGGER.debug("Rescheduling "
            + affectedCloudlets.size()
            + " cloudlets took (s): "
            + brokerTime);
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

    public List<Cloudlet> getJobsFinishedThisTimestep() {
        return jobsFinishedThisTimestep;
    }

    public double clock() {
        return cloudSimPlus.clock();
    }

    public long getNumberOfFutureEvents() {
        return cloudSimPlus.getNumberOfFutureEvents(simEvent -> true);
    }

    public long getWaitingJobsCount() {
        return submittedJobs.parallelStream()
            .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
            .count();
    }

    public double getRunningCost() {
        return vmCost.getVMCostPerIteration(clock());
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

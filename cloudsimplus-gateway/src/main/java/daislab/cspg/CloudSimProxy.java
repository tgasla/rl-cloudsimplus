package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.brokers.DatacenterBroker;
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
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.listeners.EventListener;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.IntStream;
import java.util.PriorityQueue;

public class CloudSimProxy {
    private final Logger LOGGER = LoggerFactory.getLogger(CloudSimProxy.class.getSimpleName());
    private final SimulationSettings settings;
    private final CloudSimPlus cloudSimPlus;
    private final Datacenter datacenter;
    private final DatacenterBrokerFirstFitFixed broker;
    private final VmCost vmCost;
    private final List<Cloudlet> inputJobs; // all jobs to keep track of statuses
    private final PriorityQueue<Cloudlet> jobQueue; // jobs to be submitted - sorted by arrival time
    private final Map<Long, Double> jobArrivalTimeMap; // map to keep track of arrival times
    private List<Double> jobsFinishedWaitTimeLastTimestep;
    // private List<Double> jobsFinishedWaitTimes;
    private int vmsCreated;
    private boolean firstStep;

    /**
     * Constructs a new CloudSimProxy instance with the specified simulation settings and input
     * jobs.
     *
     * @param settings the simulation settings to be used
     * @param inputJobs the list of Cloudlet jobs to be processed
     */
    public CloudSimProxy(final SimulationSettings settings, final List<Cloudlet> inputJobs) {
        this.settings = settings;
        this.inputJobs = new ArrayList<>(inputJobs);
        jobQueue = new PriorityQueue<>(inputJobs.size(),
                (c1, c2) -> Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        jobQueue.addAll(inputJobs);
        jobArrivalTimeMap = jobQueue.stream()
                .collect(Collectors.toMap(Cloudlet::getId, Cloudlet::getSubmissionDelay));
        cloudSimPlus = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        broker = new DatacenterBrokerFirstFitFixed(cloudSimPlus);
        broker.setShutdownWhenIdle(false); // important to keep the broker running
        // no need because set in createVm()
        // broker.setVmDestructionDelay(2 * settings.getMinTimeBetweenEvents());
        datacenter = createDatacenter();
        vmCost = new VmCost(settings);
        jobsFinishedWaitTimeLastTimestep = new ArrayList<>();
        // jobsFinishedWaitTimes = new ArrayList<>();
        vmsCreated = 0;
        firstStep = true;

        submitInitialVmList();
        initializeJobListeners();
        ensureAllJobsCompleteBeforeSimulationEnds();
        cloudSimPlus.startSync();

        // initialize the simulation to allow the datacenter to be created
        proceedClockTo(settings.getMinTimeBetweenEvents());
    }

    /**
     * Retrieves the number of virtual machine (VM) cores based on the specified VM type.
     *
     * @param type the type of the VM for which the core count is to be retrieved.
     * @return the number of VM cores for the specified type, calculated as the product of the small
     *         VM processing elements (PEs) and the size multiplier for the given type.
     */
    public int getVmCoreCountByType(final String type) {
        return settings.getSmallVmPes() * settings.getSizeMultiplier(type);
    }

    /**
     * Submits the initial list of VMs to the broker.
     * <p>
     * This method creates an initial list of VMs based on the VM types and counts specified in the
     * settings. Each VM is assigned a description corresponding to its type. The VMs are then added
     * to the cost tracking list and submitted to the broker.
     * </p>
     */
    private void submitInitialVmList() {
        List<Vm> initialVmList = new ArrayList<>();
        for (int i = 0; i < settings.VM_TYPES.length; i++) {
            String vmType = settings.VM_TYPES[i];
            List<Vm> vmList = createVmList(settings.getInitialVmCounts()[i], vmType);
            initialVmList.forEach(v -> v.setDescription(vmType));
            initialVmList.addAll(vmList);
        }
        initialVmList.forEach(v -> vmCost.addNewVmToList(v));
        broker.submitVmList(initialVmList);
    }

    /**
     * Initializes job listeners for each job in the inputJobs collection. This method adds both
     * start and finish listeners to each job.
     */
    private void initializeJobListeners() {
        inputJobs.forEach(this::addOnStartListener);
        inputJobs.forEach(this::addOnFinishListener);
    }

    /**
     * Ensures that all jobs are completed before the simulation ends. This method sets up an event
     * listener that checks if there are unfinished jobs when there is only one future event left.
     * If there are unfinished jobs, it sends an empty event to keep the simulation running.
     */
    private void ensureAllJobsCompleteBeforeSimulationEnds() {
        double interval = settings.getTimestepInterval();
        cloudSimPlus.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && hasUnfinishedJobs()) {
                LOGGER.trace("Jobs not finished. Sending empty event to keep simulation running.");
                cloudSimPlus.send(datacenter, datacenter, interval, CloudSimTag.NONE, null);
            }
        });
    }

    /**
     * Creates a Datacenter with a list of hosts and a VM allocation policy.
     * 
     * @return a new instance of {@link DatacenterSimple} initialized with the specified hosts and
     *         VM allocation policy.
     */
    private Datacenter createDatacenter() {
        List<Host> hostList = createHostList();
        LOGGER.debug("Creating datacenter");
        final VmAllocationPolicy vmAllocationPolicy = defineVmAllocationPolicy();

        return new DatacenterSimple(cloudSimPlus, hostList, vmAllocationPolicy);
    }

    private VmAllocationPolicy defineVmAllocationPolicy() {
        return switch (settings.getVmAllocationPolicy()) {
            case "rl", "fromfile" -> new VmAllocationPolicyCustom();
            case "heuristic" -> defineHeuristicVmAllocationPolicy();
            default -> throw new IllegalArgumentException(
                    "Unknown VM allocation policy: " + settings.getVmAllocationPolicy());
        };
    }

    private VmAllocationPolicy defineHeuristicVmAllocationPolicy() {
        return switch (settings.getAlgorithm()) {
            case "minimize-makespan", "minimize-unutilization", "minimize-energy" -> new VmAllocationPolicyBestFit();
            default -> throw new IllegalArgumentException(
                    "Unknown VM allocation policy: " + settings.getVmAllocationPolicy());
        };
    }

    /**
     * Creates a list of hosts based on the settings provided.
     * 
     * @return A list of {@link Host} objects.
     * 
     *         The method initializes a list of hosts with the following properties: - RAM,
     *         bandwidth, and storage are set according to the settings. - Each host is assigned a
     *         list of processing elements (PEs) created by the {@code createPeList()} method. -
     *         Each host is configured with simple resource provisioners for RAM and bandwidth. -
     *         Each host uses a time-shared VM scheduler.
     */
    private List<Host> createHostList() {
        List<Host> hostList = new ArrayList<>();
        final long hostRam = settings.getHostRam();
        final long hostBw = settings.getHostBw();
        final long hostStorage = settings.getHostStorage();

        for (int i = 0; i < settings.getHostsCount(); i++) {
            List<Pe> peList = createPeList();
            Host host = new HostWithoutCreatedList(hostRam, hostBw, hostStorage, peList)
                    .setRamProvisioner(new ResourceProvisionerSimple())
                    .setBwProvisioner(new ResourceProvisionerSimple())
                    .setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(host);
        }
        return hostList;
    }

    /**
     * Creates a list of virtual machines (VMs) of a specified type.
     *
     * @param vmCount the number of VMs to create
     * @param type the type of VMs to create
     * @return a list of created VMs
     */
    private List<Vm> createVmList(final int vmCount, final String type) {
        List<Vm> vmList = new ArrayList<>(vmCount);
        for (int i = 0; i < vmCount; i++) {
            vmList.add(createVm(type));
        }

        return vmList;
    }

    /**
     * Creates a new virtual machine (VM) based on the specified type.
     *
     * @param type the type of the VM to be created, which determines the size multiplier.
     * @return the newly created VM instance.
     *
     *         The VM is configured with the following properties: - Processing elements (PEs) based
     *         on the size multiplier. - RAM, bandwidth, and storage size based on the size
     *         multiplier. - An optimized cloudlet scheduler. - A shutdown delay as specified in the
     *         settings. - A submission delay (startup delay) as specified in the settings.
     *
     *         The VM is also added to the VM cost tracking list.
     */
    private Vm createVm(final String type) {
        int sizeMultiplier = settings.getSizeMultiplier(type);

        Vm vm = new VmSimple(vmsCreated++, settings.getHostPeMips(),
                settings.getSmallVmPes() * sizeMultiplier);
        vm.setRam(settings.getSmallVmRam() * sizeMultiplier).setBw(settings.getSmallVmBw())
                .setSize(settings.getSmallVmStorage())
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setShutDownDelay(settings.getVmShutdownDelay());

        // assuming average startup delay is 56s as in 10.48550/arXiv.2107.03467
        // submissiondelay or vm boot up delay
        vm.setSubmissionDelay(settings.getVmStartupDelay());
        vmCost.addNewVmToList(vm);
        return vm;
    }

    /**
     * Creates a list of Processing Elements (PEs) for a host. Each PE is initialized with a
     * specified MIPS (Million Instructions Per Second) capacity and a simple PE provisioner.
     *
     * @return a list of PEs configured according to the host settings.
     */
    private List<Pe> createPeList() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < settings.getHostPes(); i++) {
            peList.add(new PeSimple(settings.getHostPeMips(), new PeProvisionerSimple()));
        }

        return peList;
    }

    /**
     * Advances the simulation clock to the specified target time. This method runs the simulation
     * in increments until the target time is reached or the maximum number of iterations is
     * exceeded to prevent an infinite loop.
     *
     * @param targetTime The target time to advance the simulation clock to.
     */
    private void proceedClockTo(final double targetTime) {
        double adjustedInterval = targetTime - clock();
        int maxIterations = 1000; // Safety check to prevent infinite loop
        int iterations = 0;

        LOGGER.info("{}: Proceeding clock to {}", clock(), targetTime);

        // Run the simulation until the target time is reached
        while (cloudSimPlus.runFor(adjustedInterval) < targetTime) {
            // Calculate the remaining time to the target
            adjustedInterval = targetTime - clock();
            // Use the minimum time between events if the remaining time is non-positive
            adjustedInterval =
                    adjustedInterval <= 0 ? settings.getMinTimeBetweenEvents() : adjustedInterval;

            // Increment the iteration counter and break if it exceeds the maximum allowed
            // iterations
            if (++iterations >= maxIterations) {
                LOGGER.warn(
                        "Exceeded maximum iterations in runForInternal. Breaking the loop to prevent infinite loop.");
                break;
            }
        }
    }

    int calculateJobCoresWaiting() {
        final double targetTime = calculateTargetTime();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        int totalCoresRequired = coresRequiredForJobs(jobsToSubmitList);

        return totalCoresRequired;
    }

    long calculateMaxJobCoresNeeded() {
        final double targetTime = calculateTargetTime();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        // Find the maximum cores required by any single job
        return jobsToSubmitList.stream().mapToLong(Cloudlet::getPesNumber).max().orElse(0);
    }

    private int coresRequiredForJobs(List<Cloudlet> jobsToSubmitList) {
        return (int) jobsToSubmitList.stream().mapToLong(Cloudlet::getPesNumber).sum();
    }

    private boolean isJobWithCoresWaiting(final long cores) {
        final double targetTime = calculateTargetTime();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        return jobsToSubmitList.stream().anyMatch(job -> job.getPesNumber() == cores);
    }

    private boolean isVmWithCoresRunning(final long cores) {
        return broker.getVmExecList().stream().anyMatch(vm -> vm.getPesNumber() == cores);
    }

    private void executeMinimizeMakespanAction() {
        long maxCoresNeeded = calculateMaxJobCoresNeeded();
        long maxfreeCoresOnSameVm = getMaxFreeVmCores();

        // Check if any VM has at least the required cores
        boolean vmAvailable = maxfreeCoresOnSameVm >= maxCoresNeeded;

        if (!vmAvailable && maxCoresNeeded > 0) {
            // Create a new VM with the required number of cores
            List<Vm> vmList = createSingleVm(calculateTargetTime(), maxCoresNeeded);
            broker.submitVmList(vmList);
        } else {
            destroyLargestIdleVm(); // Optionally handle idle VMs
        }
    }

    private void executeMinimizeUnutilizationAction() {
        List<Vm> vmList = new ArrayList<>();
        if (isJobWithCoresWaiting(2) && !isVmWithCoresRunning(2)) {
            vmList = createSingleVm(calculateTargetTime(), 2);
            broker.submitVmList(vmList);
        } else if (isJobWithCoresWaiting(4) && !isVmWithCoresRunning(4)) {
            vmList = createSingleVm(calculateTargetTime(), 4);
            broker.submitVmList(vmList);
        } else if (isJobWithCoresWaiting(8) && !isVmWithCoresRunning(8)) {
            vmList = createSingleVm(calculateTargetTime(), 8);
            broker.submitVmList(vmList);
        } else {
            destroyLargestIdleVm();
        }
    }

    private void executeMinimizeEnergyAction() {
        // have only one VM running in total
        int coresNeeded = calculateJobCoresWaiting();
        final long smallVmCores = getVmCoreCountByType(settings.VM_TYPES[0]);
        final long mediumVmCores = getVmCoreCountByType(settings.VM_TYPES[1]);
        final long largeVmCores =
                getVmCoreCountByType(settings.VM_TYPES[settings.VM_TYPES.length - 1]);
        if (coresNeeded >= largeVmCores && !isVmWithCoresRunning(largeVmCores)) {
            List<Vm> vmList = createSingleVm(calculateTargetTime(), largeVmCores);
            broker.submitVmList(vmList);
        } else if (coresNeeded >= largeVmCores && isVmWithCoresRunning(largeVmCores)) {
        } else if (coresNeeded >= mediumVmCores && isVmWithCoresRunning(largeVmCores)) {
            destroyLargestIdleVm();
        } else if (coresNeeded >= mediumVmCores && !isVmWithCoresRunning(mediumVmCores)) {
            List<Vm> vmList = createSingleVm(calculateTargetTime(), mediumVmCores);
            broker.submitVmList(vmList);
        } else if (coresNeeded >= mediumVmCores && isVmWithCoresRunning(mediumVmCores)) {
        } else if (coresNeeded >= smallVmCores && isVmWithCoresRunning(mediumVmCores)) {
            destroyLargestIdleVm();
        } else if (coresNeeded >= smallVmCores && !isVmWithCoresRunning(smallVmCores)) {
            List<Vm> vmList = createSingleVm(calculateTargetTime(), smallVmCores);
            broker.submitVmList(vmList);
        } else if (coresNeeded >= smallVmCores && isVmWithCoresRunning(smallVmCores)) {
        } else {
            destroyLargestIdleVm();
        }
    }

    public boolean executeHeuristicAction() {
        if (settings.getAlgorithm().equals("minimize-makespan")) {
            executeMinimizeMakespanAction();
        } else if (settings.getAlgorithm().equals("minimize-unutilization")) {
            executeMinimizeUnutilizationAction();
        } else if (settings.getAlgorithm().equals("minimize-energy")) {
            executeMinimizeEnergyAction();
        }
        return true;

        // final int totalCoresRequired = calculateJobCoresWaiting();
        // if (totalCoresRequired == 0) {
        // destroyLargestIdleVm();
        // return true;
        // }
        // int totalFreeVmCores = getTotalFreeVmCores();
        // LOGGER.info("{}: Cores required by next batch of arriving jobs: {}", clock(),
        // totalCoresRequired);
        // LOGGER.info("{}: Total free VM cores: {}", clock(), totalFreeVmCores);
        // int coresNeeded = totalCoresRequired - totalFreeVmCores;
        // if (coresNeeded > 0) {
        // // List<Vm> vmList = createRequiredVms(targetTime, coresNeeded);
        // List<Vm> vmList = createSingleVm(calculateTargetTime(), coresNeeded);
        // // can also add a submission delay here: settings.getVmStartupDelay()
        // // not needed because it is done in the createVm() function
        // broker.submitVmList(vmList);
        // } else {
        // destroyLargestIdleVm();
        // }
        // return true;
    }

    double calculateTargetTime() {
        // if first step we have already done 0.1 and we need to finish the first step
        // at 1
        // else, just add 1 to the current time
        final double targetTime;
        if (firstStep) {
            targetTime = settings.getTimestepInterval();
        } else {
            targetTime = clock() + settings.getTimestepInterval();
        }
        return targetTime;
    }

    public void runOneTimestep() {
        final double targetTime = calculateTargetTime();
        ensureSimulationIsRunning();
        jobsFinishedWaitTimeLastTimestep.clear();
        List<Cloudlet> jobsToSubmitList = getJobsToSubmitAtThisTimestep(targetTime);
        clearListsIfNeeded();
        proceedClockTo(targetTime);
        tryToSubmitJobs(jobsToSubmitList);
        if (shouldPrintStats()) {
            printStats();
        }
        if (firstStep) {
            firstStep = false;
        }
        LOGGER.info("VMs running: {}", broker.getVmExecList().size());
    }

    private void destroyLargestIdleVm() {
        List<Vm> idleVms = broker.getVmExecList().stream()
                .filter(vm -> vm.getCloudletScheduler().isEmpty()).collect(Collectors.toList());

        idleVms.stream().max(Comparator.comparingLong(Vm::getPesNumber)).ifPresent(largestVm -> {
            cloudSimPlus.send(datacenter, datacenter, 0, CloudSimTag.VM_DESTROY, largestVm);
            LOGGER.info("No jobs to submit, destroying the largest idle VM");
            LOGGER.info("VMs running: {}", broker.getVmExecList().size());
        });

        if (idleVms.isEmpty()) {
            LOGGER.info("No idle VMs available for destruction.");
        }
    }

    /**
     * Ensures that the simulation is currently running. If the simulation is not running, it throws
     * an IllegalStateException.
     *
     * @throws IllegalStateException if the simulation is not running.
     */
    private void ensureSimulationIsRunning() {
        if (!isRunning()) {
            throw new IllegalStateException(
                    "Simulation is not running. Please reset or create a new one!");
        }
    }

    /**
     * Clears the lists of created cloudlets and VMs if needed to prevent OutOfMemoryError (OOM).
     * The lists can grow to large sizes as cloudlets are re-scheduled when VMs are terminated. This
     * method checks a setting to determine if the lists should be cleared. It is safe to clear
     * these lists in the current environment because they are only used in CloudSimPlus when a VM
     * is being upscaled, which is not performed in this environment.
     */
    private void clearListsIfNeeded() {
        if (settings.isClearCreatedLists()) {
            broker.getCloudletCreatedList().clear();
            // broker.getVmCreatedList().clear();
        }
    }

    private boolean shouldPrintStats() {
        return ((int) Math.round(clock()) % 1 == 0) || !isRunning();
    }

    /**
     * Logs various statistics about the jobs and VMs in the simulation.
     * <ul>
     * <li>Total number of VMs created.</li>
     * <li>Number of VMs currently running.</li>
     * <li>Total number of jobs.</li>
     * <li>Count of jobs by their status.</li>
     * <li>Number of jobs that have arrived.</li>
     * <li>Number of jobs that have arrived but are not yet running.</li>
     * </ul>
     */
    public void printStats() {
        // Contradictory to the previous functions which are called before the clock is
        // procceded,
        // this function is called after the clock is procceded. So, we need to
        // calculate the start
        // time of the timestep (instead of the target time) to get the correct
        // statistics. This is
        // done because this function also calls getArrivedJobsCount() which is also
        // called in
        // WrappedSimulation.java:calculateReward() function after the clock is
        // procceded.
        final double startTime = clock() - settings.getTimestepInterval();

        LOGGER.info("[{} - {}]: All jobs: {} ", startTime, clock(), inputJobs.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : inputJobs) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for (Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info("[{} - {}]: {}: {}", startTime, clock(), e.getKey().toString(),
                    e.getValue());
        }

        LOGGER.info("[{} - {}]: Jobs arrived: {}", startTime, clock(), getArrivedJobsCount());
    }

    private List<Vm> createSingleVm(final double targetTime, final long coresNeeded) {
        final int vmTypesCount = settings.VM_TYPES.length;
        final List<Vm> vmList = new ArrayList<>();
        final double startTime = targetTime - settings.getTimestepInterval();

        int vmTypeIndex = vmTypesCount - 1;
        // generous - creates the smallest VM type that can fulfill the cores needed
        for (int i = 0; i < vmTypesCount; i++) {
            if (coresNeeded <= getVmCoreCountByType(settings.VM_TYPES[i])) {
                vmTypeIndex = i;
                break;
            }
        }

        if (vmTypeIndex == -1) {
            LOGGER.error("[{} - {}]: No VM type can fulfill {} cores needed", startTime, targetTime,
                    coresNeeded);
            return vmList; // Return empty list if no VM type is suitable
        }

        final String vmType = settings.VM_TYPES[vmTypeIndex];
        LOGGER.info("[{} - {}]: {} VM cores are needed, will create 1 {} VM", startTime, targetTime,
                coresNeeded, vmType);
        vmList.add(createVm(vmType).setDescription(vmType));

        return vmList;
    }

    /**
     * Checks if any VM (Virtual Machine) in the broker's execution list is suitable for the given
     * cloudlet.
     *
     * @param cloudlet the cloudlet to be checked for suitability against the VMs.
     * @return {@code true} if there is at least one VM suitable for the cloudlet, {@code false}
     *         otherwise.
     */
    private boolean isAnyVmSuitableForCloudlet(Cloudlet cloudlet) {
        return broker.getVmExecList().stream().anyMatch(vm -> vm.isSuitableForCloudlet(cloudlet));
    }

    /**
     * Adds an event listener to the specified Cloudlet that triggers when the Cloudlet starts
     * running. The listener logs a debug message with the Cloudlet ID, VM ID, and the current
     * simulation time.
     *
     * @param cloudlet the Cloudlet to which the start listener will be added
     */
    private void addOnStartListener(Cloudlet cloudlet) {
        cloudlet.addOnStartListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug("Cloudlet: {} started running on VM {} at {} ", cloudlet.getId(),
                        cloudlet.getVm().getId(), clock());
            }
        });
    }

    /**
     * Adds an on-finish listener to the specified Cloudlet. The listener logs detailed information
     * about the Cloudlet's execution and calculates the wait time for the Cloudlet upon its
     * completion.
     *
     * @param cloudlet the Cloudlet to which the on-finish listener will be added
     */
    private void addOnFinishListener(Cloudlet cloudlet) {
        cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug(
                        "{}: Cloudlet: {} that was running on vm {} (runs {} cloudlets) on host {} (runs {} vms) finished at {} with total execution time {}",
                        clock(), cloudlet.getId(), cloudlet.getVm().getId(),
                        cloudlet.getVm().getCloudletScheduler().getCloudletExecList().size(),
                        cloudlet.getVm().getHost(), cloudlet.getVm().getHost().getVmList().size(),
                        clock(), cloudlet.getTotalExecutionTime());
                final double waitTime =
                        cloudlet.getStartTime() - jobArrivalTimeMap.get(cloudlet.getId());
                jobsFinishedWaitTimeLastTimestep.add(waitTime);
                // jobsFinishedWaitTimes.add(waitTime);
                LOGGER.debug("{}: cloudletWaitTime: {}", clock(), waitTime);
            }
        });
    }

    /**
     * Retrieves a list of Cloudlets that are ready to be submitted at the current timestep. The
     * Cloudlets are selected based on their submission delay, which must be less than or equal to
     * the specified target time.
     *
     * @param targetTime The target time to retrieve Cloudlets for submission.
     * @return A list of Cloudlets that are ready to be submitted at the specified target time.
     */
    List<Cloudlet> getJobsToSubmitAtThisTimestep(final double targetTime) {
        final List<Cloudlet> jobsToSubmit =
                jobQueue.stream().takeWhile(cloudlet -> cloudlet.getSubmissionDelay() <= targetTime)
                        .collect(Collectors.toList());
        return jobsToSubmit;
    }

    private int getTotalFreeVmCores() {
        int totalFreeCores = (int) Stream
                .concat(broker.getVmExecList().stream(), broker.getVmWaitingList().stream())
                .mapToLong(vm -> vm.getExpectedFreePesNumber()).sum()
                - (int) broker.getCloudletWaitingList().stream().mapToLong(Cloudlet::getPesNumber)
                        .sum();
        return Math.max(totalFreeCores, 0);
    }

    private long getMaxFreeVmCores() {
        return broker.getVmExecList().stream().mapToLong(vm -> vm.getExpectedFreePesNumber()).max()
                .orElse(0);
    }

    /**
     * Attempts to submit a list of cloudlets (jobs) to the cloud infrastructure.
     * 
     * This method filters the provided list of cloudlets to identify those that can be submitted
     * based on the suitability of available VMs and the submission delay.
     * 
     * @param cloudletList The list of cloudlets to be considered for submission.
     */
    private void tryToSubmitJobs(final List<Cloudlet> cloudletList) {
        final List<Cloudlet> jobsToSubmit = new ArrayList<>();
        final double startTime = clock() - settings.getTimestepInterval();

        LOGGER.info("[{} - {}]: Will try to submit {} jobs", startTime, clock(),
                cloudletList.size());
        LOGGER.info("[{} - {}]: VMs created: {}", startTime, clock(),
                broker.getVmCreatedList().size());
        LOGGER.info("[{} - {}]: VMs running: {}", startTime, clock(),
                broker.getVmExecList().size());
        for (Cloudlet cloudlet : cloudletList) {

            // Do not schedule cloudlet if there are no suitable vms to run it
            if (!isAnyVmSuitableForCloudlet(cloudlet)) {
                LOGGER.debug("[{} - {}]: Could not submit job {}, no suitable vm found", startTime,
                        clock(), cloudlet.getId());
                continue;
            }
            // here we calculate how much time the job needs to be submitted
            cloudlet.setSubmissionDelay(Math.max(cloudlet.getSubmissionDelay() - clock(), 0));
            jobsToSubmit.add(cloudlet);
        }

        if (!jobsToSubmit.isEmpty()) {
            jobQueue.removeAll(jobsToSubmit);
            LOGGER.info("[{} - {}]: Submitting {} jobs", startTime, clock(), jobsToSubmit.size());
            submitCloudletList(jobsToSubmit);
        }
    }

    private void submitCloudletList(final List<Cloudlet> cloudlets) {
        broker.submitCloudletList(cloudlets);
    }

    public boolean isRunning() {
        // if we don't have unfinished jobs, it doesn't make sense to execute any
        // actions
        return cloudSimPlus.isRunning() && hasUnfinishedJobs();
    }

    public void terminate() {
        cloudSimPlus.terminate();
    }

    private boolean hasUnfinishedJobs() {
        return broker.getCloudletFinishedList().size() < inputJobs.size();
    }

    public int getLastCreatedVmId() {
        return vmsCreated - 1;
    }

    public long getAllocatedCores() {
        final long reduce = broker.getVmExecList().parallelStream().map(Vm::getPesNumber)
                .reduce(Long::sum).orElse(0L);
        return reduce;
    }

    public double[] getVmCpuUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] cpuPercentUsage = new double[input.size()];

        IntStream.range(0, input.size())
                .forEach(i -> cpuPercentUsage[i] = input.get(i).getCpuPercentUtilization());

        return cpuPercentUsage;
    }

    public long getArrivedJobsCount() {
        return jobArrivalTimeMap.entrySet().parallelStream()
                .filter(entry -> entry.getValue() <= clock()).count();
    }

    // public long getArrivedJobsCountLastTimestep() {
    // double start = clock() - settings.getTimestepInterval();
    // return inputJobs.parallelStream()
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) <= clock())
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) > start).count();
    // }

    public List<Double> getFinishedJobsWaitTimeLastTimestep() {
        return jobsFinishedWaitTimeLastTimestep;
    }

    // public long getScheduledJobsCountLastTimestep() {
    // double start = clock() - settings.getTimestepInterval();
    // return inputJobs.parallelStream()
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) <= clock())
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) > start)
    // .filter(cloudlet -> (cloudlet.getStatus().equals(Cloudlet.Status.READY)
    // || cloudlet.getStatus().equals(Cloudlet.Status.QUEUED)
    // || cloudlet.getStatus().equals(Cloudlet.Status.INEXEC)
    // || cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)))
    // .count();
    // }

    public long getNotYetRunningJobsCount() {
        return inputJobs.parallelStream()
                .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) <= clock())
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

    // private long getQueuedJobsCount() {
    // return inputJobs.parallelStream()
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) <= clock())
    // .filter(cloudlet ->
    // cloudlet.getStatus().equals(Cloudlet.Status.QUEUED)).count();
    // }

    // private long getReadyJobsCount() {
    // return inputJobs.parallelStream()
    // .filter(cloudlet -> jobArrivalTimeMap.get(cloudlet.getId()) <= clock())
    // .filter(cloudlet ->
    // cloudlet.getStatus().equals(Cloudlet.Status.READY)).count();
    // }

    public long getRunningJobsCount() {
        return inputJobs.parallelStream()
                .filter(cloudlet -> cloudlet.getStatus().equals(Cloudlet.Status.INEXEC)).count();
    }

    public long getFinishedJobsCount() {
        return inputJobs.parallelStream()
                .filter(cloudlet -> cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

    public double[] getVmMemoryUsage() {
        List<Vm> input = broker.getVmExecList();
        double[] memPercentUsage = new double[input.size()];

        IntStream.range(0, input.size())
                .forEach(i -> memPercentUsage[i] = input.get(i).getRam().getPercentUtilization());

        return memPercentUsage;
    }

    public boolean addNewVm(final String type, final long hostId) {
        LOGGER.debug("Agent action: Create a {} VM on host {}", type, hostId);

        final Host host = datacenter.getHostById(hostId);

        if (host == Host.NULL) {
            LOGGER.debug("Vm creating ignored, no host with given id found");
            return false;
        }

        final Vm newVm = createVm(type);
        newVm.setDescription(type + "-" + hostId);

        if (!host.isSuitableForVm(newVm)) {
            LOGGER.debug("Vm creating ignored, host not suitable");
            LOGGER.debug("Host Vm List Size: {}", host.getVmList().size());
            LOGGER.debug("Total Vm Exec List Size: {}", broker.getVmExecList().size());
            return false;
        }

        broker.submitVm(newVm);
        LOGGER.debug("Requested VM of type: {} at host {}", type, hostId);
        return true;
    }

    // if a vm is destroyed, this method returns true, otherwise false
    public boolean removeVm(final int index) {
        List<Vm> vmExecList = broker.getVmExecList();
        final int vmCount = broker.getVmExecList().size();
        LOGGER.debug("vmExecList.size: {}", vmExecList);

        if (index >= vmCount) {
            LOGGER.warn("Can't kill VM with index {}: no such index found", index);
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
        return cloudlets.parallelStream().map(CloudletExecution::getCloudlet)
                .map(this::resetCloudlet).collect(Collectors.toList());
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

        // this internally calls Host.destroyVm
        datacenter.getVmAllocationPolicy().deallocateHostForVm(vm);
        vmCost.removeVmFromList(vm);

        // no need to clear it as it will be destroyed
        // vm.getCloudletScheduler().clear();

        LOGGER.info("{} Killing VM {} ({}), cloudlets to reschedule: {}", clock(), vm.getId(),
                vmSize, affectedCloudlets.size());
        if (!affectedCloudlets.isEmpty()) {
            rescheduleCloudlets(affectedCloudlets);
        }
    }

    private void rescheduleCloudlets(final List<Cloudlet> affectedCloudlets) {
        affectedCloudlets.forEach(cloudlet -> {
            cloudlet.setSubmissionDelay(0);
        });

        jobQueue.addAll(affectedCloudlets);
    }

    public CloudSimPlus getSimulation() {
        return cloudSimPlus;
    }

    public DatacenterBroker getBroker() {
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

    // public List<Double> getJobsFinishedWaitTimes() {
    // return jobsFinishedWaitTimes;
    // }
}

package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.IntStream;

public class CloudSimProxy {
    private final Logger LOGGER = LoggerFactory.getLogger(CloudSimProxy.class.getSimpleName());
    private final SimulationSettings settings;
    private final CloudSimPlus cloudSimPlus;
    private final Datacenter datacenter;
    private final DatacenterBrokerFirstFitFixed broker;
    private final VmCost vmCost;
    private final Map<Long, Double> originalSubmissionDelay;
    private final List<Cloudlet> inputJobs;
    private final List<Cloudlet> unsubmittedJobs;
    private final Set<Cloudlet> arrivedJobs;
    private List<Double> jobsFinishedWaitTimeLastTimestep;
    private int triedToSubmitJobCount;
    private int nextVmId;
    private int unableToSubmitJobCount;

    public CloudSimProxy(final SimulationSettings settings, final List<Cloudlet> inputJobs) {
        this.settings = settings;
        this.inputJobs = new ArrayList<>(inputJobs);
        this.unsubmittedJobs = new ArrayList<>(inputJobs);
        // arrivedJobs is Set because we don't care about order, we just need apply .size() to
        // see how many jobs have arrived (to avoid searching with streams and filtering the
        // inputJobs list and also we avoid duplicates easily
        this.arrivedJobs = new HashSet<>();
        originalSubmissionDelay = new HashMap<>();
        jobsFinishedWaitTimeLastTimestep = new ArrayList<>();
        nextVmId = 0;
        triedToSubmitJobCount = 0;
        unableToSubmitJobCount = 0;

        cloudSimPlus = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        broker = new DatacenterBrokerFirstFitFixed(cloudSimPlus);
        datacenter = createDatacenter();
        vmCost = new VmCost(settings);

        final List<Vm> smallVmList = createVmList(settings.getInitialSVmCount(), settings.SMALL);
        final List<Vm> mediumVmList = createVmList(settings.getInitialMVmCount(), settings.MEDIUM);
        final List<Vm> largeVmList = createVmList(settings.getInitialLVmCount(), settings.LARGE);
        broker.submitVmList(smallVmList);
        broker.submitVmList(mediumVmList);
        broker.submitVmList(largeVmList);

        Collections.sort(this.inputJobs,
                (c1, c2) -> Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        this.inputJobs.forEach(c -> originalSubmissionDelay.put(c.getId(), c.getSubmissionDelay()));
        this.inputJobs.forEach(c -> addOnStartListener(c));
        this.inputJobs.forEach(c -> addOnFinishListener(c));

        ensureAllJobsCompleteBeforeSimulationEnds();

        cloudSimPlus.startSync();
        runFor(settings.getMinTimeBetweenEvents());
    }

    public long getVmCoreCountByType(final String type) {
        return settings.getSmallVmPes() * settings.getSizeMultiplier(type);
    }

    // this will prevent the simulation from ending
    // while some jobs have not yet finished running
    private void ensureAllJobsCompleteBeforeSimulationEnds() {
        cloudSimPlus.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && hasUnfinishedJobs()) {
                LOGGER.debug("Jobs not finished. Sending empty event to keep simulation running.");
                sendEmptyEventAt(settings.getTimestepInterval());
            }
        });
    }

    private void sendEmptyEventAt(double time) {
        // We add NONE events to prevent the simulation from ending
        cloudSimPlus.send(datacenter, datacenter, time, CloudSimTag.NONE, null);
    }

    private Datacenter createDatacenter() {
        List<Host> hostList = new ArrayList<>();

        for (int i = 0; i < settings.getHostsCount(); i++) {
            List<Pe> peList = createPeList();

            final long hostRam = settings.getHostRam();
            final long hostBw = settings.getHostBw();
            final long hostStorage = settings.getHostStorage();
            Host host = new HostWithoutCreatedList(hostRam, hostBw, hostStorage, peList)
                    .setRamProvisioner(new ResourceProvisionerSimple())
                    .setBwProvisioner(new ResourceProvisionerSimple())
                    .setVmScheduler(new VmSchedulerTimeShared());

            hostList.add(host);
        }
        LOGGER.debug("Creating datacenter");
        return new DatacenterSimple(cloudSimPlus, hostList, new VmAllocationPolicyRl());
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
        int sizeMultiplier = settings.getSizeMultiplier(type);

        Vm vm = new VmSimple(nextVmId, settings.getHostPeMips(),
                settings.getSmallVmPes() * sizeMultiplier);
        vm.setRam(settings.getSmallVmRam() * sizeMultiplier).setBw(settings.getSmallVmBw())
                .setSize(settings.getSmallVmStorage())
                .setCloudletScheduler(new OptimizedCloudletScheduler()).setDescription(type)
                .setShutDownDelay(settings.getVmShutdownDelay());

        return vm;
    }

    private List<Pe> createPeList() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < settings.getHostPes(); i++) {
            peList.add(new PeSimple(settings.getHostPeMips(), new PeProvisionerSimple()));
        }

        return peList;
    }

    // This function proceeds simulation clock time
    private void runForInternal(final double interval, final double target) {
        double adjustedInterval = interval;
        // Run the simulation until the target time is reached
        while (cloudSimPlus.runFor(adjustedInterval) < target) {
            // Calculate the remaining time to the target
            adjustedInterval = target - clock();
            // Use the minimum time between events if the remaining time is non-positive
            adjustedInterval =
                    adjustedInterval <= 0 ? settings.getMinTimeBetweenEvents() : adjustedInterval;
        }
    }

    public void runFor(final double interval) {
        if (!isRunning()) {
            throw new IllegalStateException(
                    "The simulation is not running - " + "please reset or create a new one!");
        }

        final double target = cloudSimPlus.clock() + interval;

        jobsFinishedWaitTimeLastTimestep.clear();

        scheduleJobsUntil(target);
        LOGGER.debug("Jobs waiting: {}", getWaitingJobsCount());
        runForInternal(interval, target);

        if (shouldPrintJobStats()) {
            printJobStats();
        }

        // the size of cloudletsCreatedList grows to huge numbers
        // as we re-schedule cloudlets when VMs get killed
        // to avoid OOMing we need to clear that list
        // it is a safe operation in our environment, because that list is only used in
        // CloudSimPlus when a VM is being upscaled (we don't do that)
        if (settings.isClearCreatedCloudletList()) {
            broker.getCloudletCreatedList().clear();
        }

        String startTimeFormat = String.format("%.1f", clock() - interval);
        LOGGER.debug("runFor [{}-{}]", startTimeFormat, clock());
    }

    private boolean shouldPrintJobStats() {
        return ((int) Math.round(clock()) % 1000 == 0) || !isRunning();
    }

    public void printJobStats() {
        if (!isRunning()) {
            LOGGER.info("Simulation terminated. Jobs finished: {}/{}",
                    getBroker().getCloudletFinishedList().size(), inputJobs.size());
        }
        LOGGER.info("All jobs: " + inputJobs.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : inputJobs) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for (Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info(e.getKey().toString() + ": " + e.getValue());
        }

        // LOGGER.info("Jobs which are still queued");
        // for(Cloudlet cloudlet : inputJobs) {
        // if (Cloudlet.Status.QUEUED.equals(cloudlet.getStatus())) {
        // printCloudlet(cloudlet);
        // }
        // }
        // LOGGER.info("Jobs which are still executed");
        // for(Cloudlet cloudlet : inputJobs) {
        // if (Cloudlet.Status.INEXEC.equals(cloudlet.getStatus())) {
        // printCloudlet(cloudlet);
        // }
        // }

        // LOGGER.debug("jobs arrived:" + cloudSimProxy.getArrivedJobsCount());
        // LOGGER.debug("jobs waiting: " + cloudSimProxy.getWaitingJobsCount());
        // LOGGER.debug("jobs running: " + cloudSimProxy.getRunningJobsCount());
        // LOGGER.debug("jobs finished: " + cloudSimProxy.getFinishedJobsCount());
    }

    // private void printCloudlet(final Cloudlet cloudlet) {
    // LOGGER.info("Cloudlet: " + cloudlet.getId());
    // LOGGER.info("Number of PEs: " + cloudlet.getPesNumber());
    // LOGGER.info("Total length in MIs: " + cloudlet.getTotalLength());
    // LOGGER.info("Submission delay: " + originalSubmissionDelay.get(cloudlet.getId()));
    // LOGGER.info("Started: " + cloudlet.getStartTime());
    // final Vm vm = cloudlet.getVm();
    // LOGGER.info("VM: " + vm.getId() + "(" + vm.getDescription() + ")" + " CPU: "
    // + vm.getPesNumber() + "/" + vm.getMips() + " @ " + vm.getCpuPercentUtilization()
    // + " RAM: " + vm.getRam().getAllocatedResource() + " Start time: "
    // + vm.getStartTime() + " Stop time: " + vm.getFinishTime());
    // }

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
                LOGGER.debug("Cloudlet: {} started running on VM {} at {} ", cloudlet.getId(),
                        cloudlet.getVm().getId(), clock());
            }
        });
    }

    private void addOnFinishListener(Cloudlet cloudlet) {
        cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.debug("Cloudlet: " + cloudlet.getId() + " that was running on vm "
                        + cloudlet.getVm().getId() + " (runs "
                        + cloudlet.getVm().getCloudletScheduler().getCloudletExecList().size()
                        + " cloudlets) on host " + cloudlet.getVm().getHost() + " (runs "
                        + cloudlet.getVm().getHost().getVmList().size() + " vms) finished at "
                        + clock() + " with total execution time "
                        + cloudlet.getTotalExecutionTime());
                final double waitTime =
                        cloudlet.getStartTime() - originalSubmissionDelay.get(cloudlet.getId());
                jobsFinishedWaitTimeLastTimestep.add(waitTime);
                LOGGER.debug("cloudletWaitTime: {}", waitTime);
            }
        });
    }

    private void scheduleJobsUntil(final double target) {
        List<Cloudlet> jobsToSubmit = new ArrayList<>();
        this.triedToSubmitJobCount = 0;
        this.unableToSubmitJobCount = 0;

        for (final Iterator<Cloudlet> it = unsubmittedJobs.iterator(); it.hasNext();) {
            Cloudlet cloudlet = it.next();
            if (cloudlet.getSubmissionDelay() > target) {
                continue;
            }

            // arrivedJobs is a Set, so when jobs already arrived (and in the list) need to be
            // rescheduled, those jobs are added to the unsubmittedJobs list again for rescheduling
            // but they will not be added twice in the arrivedJobs Set
            this.arrivedJobs.add(cloudlet);

            triedToSubmitJobCount++;
            // Do not schedule cloudlet if there are no suitable vms to run it
            if (!isAnyVmSuitableForCloudlet(cloudlet)) {
                unableToSubmitJobCount++;
                // LOGGER.debug("unable to submit cloudlet. Status remains " +
                // cloudlet.getStatus());
                continue;
            }
            // here we calculate how much time the job needs to be submitted
            cloudlet.setSubmissionDelay(Math.max(cloudlet.getSubmissionDelay() - clock(), 0));
            jobsToSubmit.add(cloudlet);
            it.remove();
        }

        if (!jobsToSubmit.isEmpty()) {
            submitCloudletsList(jobsToSubmit);
        }

        if (this.unableToSubmitJobCount > 0) {
            LOGGER.debug("Unable to submit {} jobs: no vm available", this.unableToSubmitJobCount);
        }
    }

    private void submitCloudletsList(final List<Cloudlet> jobsToSubmit) {
        LOGGER.debug("Submitting: {} jobs", jobsToSubmit.size());
        broker.submitCloudletList(jobsToSubmit);

        // we immediately clear up that list because it is not really
        // used anywhere but traversing it takes a lot of time
        // No need because the history of this list is already disabled by default
        // see: https://javadoc.io/doc/org.cloudsimplus/cloudsim-plus/latest/org/cloudsimplus/
        // schedulers/cloudlet/CloudletSchedulerAbstract.html#enableCloudletSubmittedList()
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

    // unoptimal
    // public long getArrivedJobsCount() {
    // return inputJobs.parallelStream()
    // .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
    // .count();
    // }

    public long getArrivedJobsCount() {
        return this.arrivedJobs.size();
    }

    public long getArrivedJobsCountLastTimestep() {
        double start = clock() - settings.getTimestepInterval();
        return inputJobs.parallelStream()
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) > start).count();
    }

    public List<Double> getFinishedJobsWaitTimeLastTimestep() {
        return jobsFinishedWaitTimeLastTimestep;
    }

    public int getUnableToSubmitJobCount() {
        return this.unableToSubmitJobCount;
    }

    public int getTriedToSubmitJobCount() {
        return this.triedToSubmitJobCount;
    }

    public long getWaitingJobsCountLastTimestep() {
        double start = clock() - settings.getTimestepInterval();
        return inputJobs.parallelStream()
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) > start)
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

    public long getScheduledJobsCountLastTimestep() {
        double start = clock() - settings.getTimestepInterval();
        return inputJobs.parallelStream()
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) > start)
                .filter(cloudlet -> (cloudlet.getStatus().equals(Cloudlet.Status.READY)
                        || cloudlet.getStatus().equals(Cloudlet.Status.QUEUED)
                        || cloudlet.getStatus().equals(Cloudlet.Status.INEXEC)
                        || cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)))
                .count();
    }

    public long getWaitingJobsCount() {
        return inputJobs.parallelStream()
                .filter(cloudlet -> originalSubmissionDelay.get(cloudlet.getId()) <= clock())
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

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
            // LOGGER.debug("Host MIPS:" + host.getVmScheduler().getTotalAvailableMips());
            LOGGER.debug("Host Vm List Size: {}", host.getVmList().size());
            LOGGER.debug("Total Vm Exec List Size: {}", broker.getVmExecList().size());
            return false;
        }

        vmCost.addNewVmToList(newVm);
        nextVmId++;

        // assuming average startup delay is 56s as in 10.48550/arXiv.2107.03467
        final double delay = settings.getVmStartupDelay();
        // submissiondelay or vm boot up delay
        newVm.setSubmissionDelay(delay);
        broker.submitVm(newVm);
        LOGGER.debug("Requested VM of type: {}, delay: {}", type, delay);
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

        // this internally calls destroyVm
        datacenter.getVmAllocationPolicy().deallocateHostForVm(vm);

        vm.getCloudletScheduler().clear();

        LOGGER.debug("Killing VM {} ({}), cloudlets to reschedule: {}" + vm.getId(), vmSize,
                affectedCloudlets.size());
        if (!affectedCloudlets.isEmpty()) {
            rescheduleCloudlets(affectedCloudlets);
        }
    }

    private void rescheduleCloudlets(final List<Cloudlet> affectedCloudlets) {
        affectedCloudlets.forEach(cloudlet -> {
            cloudlet.setSubmissionDelay(0);
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
}

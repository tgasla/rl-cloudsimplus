package daislab.csg;

import static org.apache.commons.lang3.SystemUtils.getEnvironmentVariable;

public class SimulationSettings {

    private final double vmRunningHourlyCost;
    private final long hostPeMips;
    private final long hostBw;
    private final long hostRam;
    private final long hostSize;
    private final int hostPeCnt;
    private final double queueWaitPenalty;
    private final long datacenterHostsCnt;
    private final long basicVmRam;
    private final long basicVmPeCount;
    private final double vmShutdownDelay;
    private final long maxVmsPerSize;
    private final boolean printJobsPeriodically;
    private final boolean payingForTheFullHour;
    private final boolean storeCreatedCloudletsDatacenterBroker;

    public SimulationSettings() {
        // Host size is big enough to host a m5a.2xlarge VM
        vmRunningHourlyCost = Double.parseDouble(getEnvironmentVariable("VM_RUNNING_HOURLY_COST", "0.2"));
        hostPeMips = Long.parseLong(getEnvironmentVariable("HOST_PE_MIPS", "10000"));
        hostBw = Long.parseLong(getEnvironmentVariable("HOST_BW", "50000"));
        hostRam = Long.parseLong(getEnvironmentVariable("HOST_RAM", "65536"));
        hostSize = Long.parseLong(getEnvironmentVariable("HOST_SIZE", "16000"));
        hostPeCnt = Integer.parseInt(getEnvironmentVariable("HOST_PE_CNT", "14"));
        queueWaitPenalty = Double.parseDouble(getEnvironmentVariable("QUEUE_WAIT_PENALTY", "0.00001"));
        datacenterHostsCnt = Long.parseLong(getEnvironmentVariable("DATACENTER_HOSTS_CNT", "50"));
        basicVmRam = Long.parseLong(getEnvironmentVariable("BASIC_VM_RAM", "8192"));
        basicVmPeCount = Long.parseLong(getEnvironmentVariable("BASIC_VM_PE_CNT", "2"));
        vmShutdownDelay = Double.parseDouble(getEnvironmentVariable("VM_SHUTDOWN_DELAY", "0"));

        // we can have as many VMs as the number of hosts, as every host can have 1 small, 1 medium and 1 large Vm
        maxVmsPerSize = Long.parseLong(getEnvironmentVariable("MAX_VMS_PER_SIZE", "50"));
        printJobsPeriodically = Boolean.parseBoolean(getEnvironmentVariable("PRINT_JOBS_PERIODICALLY", "false"));
        payingForTheFullHour = Boolean.parseBoolean(getEnvironmentVariable("PAYING_FOR_THE_FULL_HOUR", "false"));
        storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(getEnvironmentVariable("STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));
    }

    @Override
    public String toString() {
        return "SimulationSettings{" +
                "\nvmRunningHourlyCost=" + vmRunningHourlyCost +
                "\n, hostPeMips=" + hostPeMips +
                "\n, hostBw=" + hostBw +
                "\n, hostRam=" + hostRam +
                "\n, hostSize=" + hostSize +
                "\n, hostPeCnt=" + hostPeCnt +
                "\n, queueWaitPenalty=" + queueWaitPenalty +
                "\n, datacenterHostsCnt=" + datacenterHostsCnt +
                "\n, basicVmRam=" + basicVmRam +
                "\n, basicVmPeCount=" + basicVmPeCount +
                "\n, maxVmsPerSize=" + maxVmsPerSize +
                "\n, printJobsPeriodically=" + printJobsPeriodically +
                "\n, payingForTheFullHour=" + payingForTheFullHour +
                "\n}";
    }

    public double getVmRunningHourlyCost() {
        return vmRunningHourlyCost;
    }

    public long getHostPeMips() {
        return hostPeMips;
    }

    public long getHostBw() {
        return hostBw;
    }

    public long getHostRam() {
        return hostRam;
    }

    public long getHostSize() {
        return hostSize;
    }

    public int getHostPeCnt() {
        return hostPeCnt;
    }

    public double getQueueWaitPenalty() {
        return queueWaitPenalty;
    }

    public long getDatacenterHostsCnt() {
        return datacenterHostsCnt;
    }

    public long getDatacenterCores() {
        return getDatacenterHostsCnt() * hostPeCnt;
    }

    public long getBasicVmPeCnt() {
        return this.basicVmPeCount;
    }

    public double getVmShutdownDelay() {
        return this.vmShutdownDelay;
    }

    public long getBasicVmSize() {
        return this.getHostSize() / 4;
    }

    public long getBasicVmBw() {
        return this.getHostBw() / 4;
    }

    public long getBasicVmRam() {
        return this.basicVmRam;
    }

    public long getAvailableCores() {
        // we can have 2 cores for a small VM, 4 for Medium and 8 for a large one
        return getMaxVmsPerSize() 
            * (getBasicVmPeCnt() + 
            getBasicVmPeCnt() * 2 +
            getBasicVmPeCnt() * 4);
    }

    public long getMaxVmsPerSize() {
        return maxVmsPerSize;
    }

    public boolean getPrintJobsPeriodically() {
        return printJobsPeriodically;
    }

    public boolean isPayingForTheFullHour() {
        return payingForTheFullHour;
    }

    public boolean isStoreCreatedCloudletsDatacenterBroker() {
        return storeCreatedCloudletsDatacenterBroker;
    }
}

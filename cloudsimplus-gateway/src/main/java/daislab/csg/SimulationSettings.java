package daislab.csg;

import java.util.Map;

import static org.apache.commons.lang3.SystemUtils.getEnvironmentVariable;

/*
 * Class to describe the simulation settings.
 * We provide two constructors.
 * 
 * The first one that takes no parameters, creates the simulation
 * by taking the settings from the environment variables.
 * If a parameter is not found as an environment variable, a default value is given.
 * 
 * The second one takes as a parameter a Map<String, String>.
 * The first string represents the parameter name
 * and the second string represents the parameter value.
*/
public class SimulationSettings {

    private final double vmRunningHourlyCost;
    private final double queueWaitPenalty;
    private final long hostSPeMips;
    private final int hostSPeCnt;
    private final long hostSRam;
    private final long hostMPeMips;
    private final int hostMPeCnt;
    private final long hostMRam;
    private final long hostLPeMips;
    private final int hostLPeCnt;
    private final long hostLRam;
    private final long hostXLPeMips;
    private final int hostXLPeCnt;
    private final long hostXLRam;
    private final long host2XLPeMips;
    private final int host2XLPeCnt;
    private final long host2XLRam;
    private final long hostBw;
    private final long hostSize;
    private final long datacenterSHostsCnt;
    private final long datacenterMHostsCnt;
    private final long datacenterLHostsCnt;
    private final long datacenterXLHostsCnt;
    private final long datacenter2XLHostsCnt;
    private final long basicVmRam;
    private final long basicVmPeCount;
    private final double vmShutdownDelay;
    private final long maxVmsPerSize;
    private final boolean printJobsPeriodically;
    private final boolean payingForTheFullHour;
    private final boolean storeCreatedCloudletsDatacenterBroker;

    // // Get SimulationSettings from environment variables,
    // //  if an environment variable is not set, a default value is given
    // public SimulationSettings() {
    //     // Host size is big enough to host a m5a.2xlarge VM
    //     vmRunningHourlyCost = Double.parseDouble(
    //             getEnvironmentVariable("VM_RUNNING_HOURLY_COST", "0.2"));
    //     hostPeMips = Long.parseLong(
    //             getEnvironmentVariable("HOST_PE_MIPS", "10000"));
    //     hostBw = Long.parseLong(
    //             getEnvironmentVariable("HOST_BW", "50000"));
    //     hostRam = Long.parseLong(
    //             getEnvironmentVariable("HOST_RAM", "65536"));
    //     hostSize = Long.parseLong(
    //             getEnvironmentVariable("HOST_SIZE", "16000"));
    //     hostPeCnt = Integer.parseInt(
    //             getEnvironmentVariable("HOST_PE_CNT", "14"));
    //     queueWaitPenalty = Double.parseDouble(
    //             getEnvironmentVariable("QUEUE_WAIT_PENALTY", "0.00001"));
    //     datacenterSHostsCnt = Long.parseLong(
    //             getEnvironmentVariable("DATACENTER_S_HOSTS_CNT", "3000"));
    //     basicVmRam = Long.parseLong(
    //             getEnvironmentVariable("BASIC_VM_RAM", "8192"));
    //     basicVmPeCount = Long.parseLong(
    //             getEnvironmentVariable("BASIC_VM_PE_CNT", "2"));
    //     vmShutdownDelay = Double.parseDouble(
    //             getEnvironmentVariable("VM_SHUTDOWN_DELAY", "0.0"));

    //     // we can have as many VMs as the number of hosts, 
    //     // as every host can have 1 small, 1 medium and 1 large Vm
    //     maxVmsPerSize = Long.parseLong(
    //             getEnvironmentVariable("MAX_VMS_PER_SIZE", "3000"));
    //     printJobsPeriodically = Boolean.parseBoolean(
    //             getEnvironmentVariable("PRINT_JOBS_PERIODICALLY", "false"));
    //     payingForTheFullHour = Boolean.parseBoolean(
    //             getEnvironmentVariable("PAYING_FOR_THE_FULL_HOUR", "false"));
    //     storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(
    //             getEnvironmentVariable(
    //             "STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));
    // }

    // Get SimulationSettings from parameters passed from the python client endpoint.
    // If an environment variable is not set, a default value is given.
    public SimulationSettings(Map<String, String> parameters) {
        vmRunningHourlyCost = Double.parseDouble(
                parameters.getOrDefault("VM_RUNNING_HOURLY_COST", "0.2"));
        queueWaitPenalty = Double.parseDouble(
                parameters.getOrDefault("QUEUE_WAIT_PENALTY", "0.00001"));
        hostSPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_S_PE_CNT", "22"));
        hostSPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_S_PE_MIPS", "110000"));
        hostSRam = Long.parseLong(
                parameters.getOrDefault("HOST_S_RAM", "128000"));
        hostMPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_M_PE_CNT", "0"));
        hostMPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_M_PE_MIPS", "0"));
        hostMRam = Long.parseLong(
                parameters.getOrDefault("HOST_M_RAM", "0"));
        hostLPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_L_PE_CNT", "0"));
        hostLPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_L_PE_MIPS", "0"));
        hostLRam = Long.parseLong(
                parameters.getOrDefault("HOST_L_RAM", "0"));
        hostXLPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_XL_PE_CNT", "0"));
        hostXLPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_XL_PE_MIPS", "0"));
        hostXLRam = Long.parseLong(
                parameters.getOrDefault("HOST_XL_RAM", "0"));
        host2XLPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_2XL_PE_CNT", "0"));
        host2XLPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_2XL_PE_MIPS", "0"));
        host2XLRam = Long.parseLong(
                parameters.getOrDefault("HOST_2XL_RAM", "0"));
        hostBw = Long.parseLong(
                parameters.getOrDefault("HOST_BW", "40000"));
        hostSize = Long.parseLong(
                parameters.getOrDefault("HOST_SIZE", "50000"));
        datacenterMHostsCnt = Long.parseLong(
                parameters.getOrDefault("DATACENTER_M_HOSTS_CNT", "0"));
        datacenterLHostsCnt = Long.parseLong(
                parameters.getOrDefault("DATACENTER_L_HOSTS_CNT", "0"));
        datacenterXLHostsCnt = Long.parseLong(
                parameters.getOrDefault("DATACENTER_XL_HOSTS_CNT", "0"));
        datacenter2XLHostsCnt = Long.parseLong(
                parameters.getOrDefault("DATACENTER_2XL_HOSTS_CNT", "0"));
        basicVmRam = Long.parseLong(
                parameters.getOrDefault("BASIC_VM_RAM", "8192"));
        basicVmPeCount = Long.parseLong(
                parameters.getOrDefault("BASIC_VM_PE_CNT", "2"));
        vmShutdownDelay = Double.parseDouble(
                parameters.getOrDefault("VM_SHUTDOWN_DELAY", "0.0"));

        // we can have as many VMs as the number of hosts, 
        // as every host can have 1 small, 1 medium and 1 large Vm
        maxVmsPerSize = Long.parseLong(
                parameters.getOrDefault("MAX_VMS_PER_SIZE", "50"));
        printJobsPeriodically = Boolean.parseBoolean(
                parameters.getOrDefault("PRINT_JOBS_PERIODICALLY", "false"));
        payingForTheFullHour = Boolean.parseBoolean(
                parameters.getOrDefault("PAYING_FOR_THE_FULL_HOUR", "false"));
        storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(
                parameters.getOrDefault(
                "STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));

        final String datacenterHostsCntStr = parameters.get("DATACENTER_HOSTS_CNT");

        if (datacenterHostsCntStr != null) {
            datacenterSHostsCnt = Long.parseLong(datacenterHostsCntStr);
        }
        else {
            datacenterSHostsCnt = Long.parseLong(
                    parameters.getOrDefault("DATACENTER_S_HOSTS_CNT", "20"));
        }
    }

    @Override
    public String toString() {
        return "SimulationSettings{" +
                "\nvmRunningHourlyCost=" + vmRunningHourlyCost +
                "\n, hostSPeCnt=" + hostSPeCnt +
                "\n, hostSPeMips=" + hostSPeMips +
                "\n, hostSRam=" + hostSRam +
                "\n, hostMPeCnt=" + hostMPeCnt +
                "\n, hostMPeMips=" + hostMPeMips +
                "\n, hostMRam=" + hostMRam +
                "\n, hostLPeCnt=" + hostLPeCnt +
                "\n, hostLPeMips=" + hostLPeMips +
                "\n, hostLRam=" + hostLRam +
                "\n, hostXLPeCnt=" + hostXLPeCnt +
                "\n, hostXLPeMips=" + hostXLPeMips +
                "\n, hostXLRam=" + hostXLRam +
                "\n, hostBw=" + hostBw +
                "\n, hostSize=" + hostSize +
                "\n, queueWaitPenalty=" + queueWaitPenalty +
                "\n, datacenterSHostsCnt=" + datacenterSHostsCnt +
                "\n, datacenterMHostsCnt=" + datacenterMHostsCnt +
                "\n, datacenterLHostsCnt=" + datacenterLHostsCnt +
                "\n, datacenterXLHostsCnt=" + datacenterXLHostsCnt +
                "\n, datacenter2XLHostsCnt=" + datacenter2XLHostsCnt +
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

    public double getQueueWaitPenalty() {
        return queueWaitPenalty;
    }

    public int getHostSPeCnt() {
        return hostSPeCnt;
    }

    public long getHostSPeMips() {
        return hostSPeMips;
    }

    public long getHostSRam() {
        return hostSRam;
    }

    public int getHostMPeCnt() {
        return hostMPeCnt;
    }

    public long getHostMPeMips() {
        return hostMPeMips;
    }

    public long getHostMRam() {
        return hostMRam;
    }

    public int getHostLPeCnt() {
        return hostLPeCnt;
    }

    public long getHostLPeMips() {
        return hostLPeMips;
    }

    public long getHostLRam() {
        return hostLRam;
    }

    public int getHostXLPeCnt() {
        return hostXLPeCnt;
    }

    public long getHostXLPeMips() {
        return hostXLPeMips;
    }

    public long getHostXLRam() {
        return hostXLRam;
    }

    public int getHost2XLPeCnt() {
        return host2XLPeCnt;
    }

    public long getHost2XLPeMips() {
        return host2XLPeMips;
    }

    public long getHost2XLRam() {
        return host2XLRam;
    }

    public long getHostBw() {
        return hostBw;
    }

    public long getHostSize() {
        return hostSize;
    }

    public long getDatacenterSHostsCnt() {
        return datacenterSHostsCnt;
    }

    public long getDatacenterMHostsCnt() {
        return datacenterMHostsCnt;
    }

    public long getDatacenterLHostsCnt() {
        return datacenterLHostsCnt;
    }

    public long getDatacenterXLHostsCnt() {
        return datacenterXLHostsCnt;
    }

    public long getDatacenter2XLHostsCnt() {
        return datacenter2XLHostsCnt;
    }

    public long getDatacenterCores() {
        return getDatacenterSHostsCnt() * getHostSPeCnt()
                + getDatacenterMHostsCnt() * getHostMPeCnt()
                + getDatacenterLHostsCnt() * getHostLPeCnt()
                + getDatacenterXLHostsCnt() * getHostXLPeCnt()
                + getDatacenter2XLHostsCnt() * getHost2XLPeCnt();
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
        // we can have 2 cores for a small VM, 
        // 2*2=4 for Medium and 4*2=8 for a large one
        return getMaxVmsPerSize() 
                * (getBasicVmPeCnt()
                + getBasicVmPeCnt() * 2
                + getBasicVmPeCnt() * 4);
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

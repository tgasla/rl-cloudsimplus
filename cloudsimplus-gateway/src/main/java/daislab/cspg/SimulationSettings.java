package daislab.cspg;

import java.util.Map;

// import static org.apache.commons.lang3.SystemUtils.getEnvironmentVariable;

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
    private final long hostPeMips;
    private final long hostBw;
    private final long hostRam;
    private final long hostSize;
    private final int hostPeCnt;
    private final double queueWaitPenalty;
    private final long datacenterHostsCnt;
    private final long basicVmRam;
    private final long basicVmPeCount;
    private final double vmStartupDelay;
    private final double vmShutdownDelay;
    private final long maxVmsPerSize;
    private final boolean printJobsPeriodically;
    private final boolean payingForTheFullHour;
    private final boolean storeCreatedCloudletsDatacenterBroker;

    // Get SimulationSettings from environment variables,
    //  if an environment variable is not set, a default value is given
//     public SimulationSettings() {
//         // Host size is big enough to host a m5a.2xlarge VM
//         vmRunningHourlyCost = Double.parseDouble(
//                 getEnvironmentVariable("VM_RUNNING_HOURLY_COST", "0.2"));
//         hostPeMips = Long.parseLong(
//                 getEnvironmentVariable("HOST_PE_MIPS", "10000"));
//         hostBw = Long.parseLong(
//                 getEnvironmentVariable("HOST_BW", "50000"));
//         hostRam = Long.parseLong(
//                 getEnvironmentVariable("HOST_RAM", "65536"));
//         hostSize = Long.parseLong(
//                 getEnvironmentVariable("HOST_SIZE", "16000"));
//         hostPeCnt = Integer.parseInt(
//                 getEnvironmentVariable("HOST_PE_CNT", "14"));
//         queueWaitPenalty = Double.parseDouble(
//                 getEnvironmentVariable("QUEUE_WAIT_PENALTY", "0.00001"));
//         datacenterHostsCnt = Long.parseLong(
//                 getEnvironmentVariable("DATACENTER_HOSTS_CNT", "3000"));
//         basicVmRam = Long.parseLong(
//                 getEnvironmentVariable("BASIC_VM_RAM", "8192"));
//         basicVmPeCount = Long.parseLong(
//                 getEnvironmentVariable("BASIC_VM_PE_CNT", "2"));
//         vmShutdownDelay = Double.parseDouble(
//                 getEnvironmentVariable("VM_SHUTDOWN_DELAY", "0"));

//         // we can have as many VMs as the number of hosts, 
//         // as every host can have 1 small, 1 medium and 1 large Vm
//         maxVmsPerSize = Long.parseLong(
//                 getEnvironmentVariable("MAX_VMS_PER_SIZE", "3000"));
//         printJobsPeriodically = Boolean.parseBoolean(
//                 getEnvironmentVariable("PRINT_JOBS_PERIODICALLY", "false"));
//         payingForTheFullHour = Boolean.parseBoolean(
//                 getEnvironmentVariable("PAYING_FOR_THE_FULL_HOUR", "false"));
//         storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(
//                 getEnvironmentVariable(
//                 "STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));
//     }

    // Get SimulationSettings from parameters
    // passed from the python client endpoint - the Gymnasium environment,
    // if an environment variable is not set, a default value is given
    // TODO: I should create a map inside the class defining the parameter names.
    // Then, get the parameters and do a get for every parameter in a loop
    // to set the parameters.
    // TODO: Maybe even better, write these parameter names and values into a file
    // so that both java and python can get these without repeating code.
    public SimulationSettings(final Map<String, String> parameters) {
        vmRunningHourlyCost = Double.parseDouble(
                parameters.getOrDefault("VM_RUNNING_HOURLY_COST", "0.2"));
        hostPeMips = Long.parseLong(
                parameters.getOrDefault("HOST_PE_MIPS", "10000"));
        hostBw = Long.parseLong(
                parameters.getOrDefault("HOST_BW", "50000"));
        hostRam = Long.parseLong(
                parameters.getOrDefault("HOST_RAM", "65536"));
        hostSize = Long.parseLong(
                parameters.getOrDefault("HOST_SIZE", "16000"));
        hostPeCnt = Integer.parseInt(
                parameters.getOrDefault("HOST_PE_CNT", "14"));
        queueWaitPenalty = Double.parseDouble(
                parameters.getOrDefault("QUEUE_WAIT_PENALTY", "0.00001"));
        datacenterHostsCnt = Long.parseLong(
                parameters.getOrDefault("DATACENTER_HOSTS_CNT", "3000"));
        basicVmRam = Long.parseLong(
                parameters.getOrDefault("BASIC_VM_RAM", "8192"));
        basicVmPeCount = Long.parseLong(
                parameters.getOrDefault("BASIC_VM_PE_CNT", "2"));
        vmStartupDelay = Double.parseDouble(
                parameters.getOrDefault("VM_STARTUP_DELAY", "0"));
        vmShutdownDelay = Double.parseDouble(
                parameters.getOrDefault("VM_SHUTDOWN_DELAY", "0"));

        // we can have as many VMs as the number of hosts, 
        // as every host can have 1 small, 1 medium and 1 large Vm
        maxVmsPerSize = Long.parseLong(
                parameters.getOrDefault("MAX_VMS_PER_SIZE", "3000"));
        printJobsPeriodically = Boolean.parseBoolean(
                parameters.getOrDefault("PRINT_JOBS_PERIODICALLY", "false"));
        payingForTheFullHour = Boolean.parseBoolean(
                parameters.getOrDefault("PAYING_FOR_THE_FULL_HOUR", "false"));
        storeCreatedCloudletsDatacenterBroker = Boolean.parseBoolean(
                parameters.getOrDefault(
                "STORE_CREATED_CLOUDLETS_DATACENTER_BROKER", "false"));
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
        return datacenterHostsCnt * hostPeCnt;
    }

    public long getBasicVmPeCnt() {
        return basicVmPeCount;
    }

    public double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public double getVmShutdownDelay() {
        return vmShutdownDelay;
    }

    public long getBasicVmSize() {
        return hostSize / 4;
    }

    public long getBasicVmBw() {
        return hostBw / 4;
    }

    public long getBasicVmRam() {
        return basicVmRam;
    }

    public long getAvailableCores() {
        // we can have 2 cores for a small VM, 
        // 4 for Medium and 8 for a large one
        return maxVmsPerSize
                * (basicVmPeCount
                + basicVmPeCount * 2
                + basicVmPeCount * 4);
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

package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.core.Simulation;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;

import java.util.List;

public class LoggingDatacenter extends DatacenterSimple {

    public LoggingDatacenter(
        final Simulation simulation,
        final List<? extends Host> hostList,
        final VmAllocationPolicy vmAllocationPolicy) {
        super(simulation, hostList, vmAllocationPolicy);
    }
}

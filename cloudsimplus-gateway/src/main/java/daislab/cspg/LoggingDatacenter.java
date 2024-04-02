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

    @Override
    protected double updateCloudletProcessing() {
        final double retVal = super.updateCloudletProcessing();

        LOGGER.debug("updateCloudletProcessing: "
                + retVal + " (if equal to Double.MAX_VALUE: "
                + Double.MAX_VALUE + " no further processing scheduled");

        return retVal;
    }
}

package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.core.Simulation;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class LoggingDatacenter extends DatacenterSimple {
    private static final Logger LOGGER
            = LoggerFactory.getLogger(LoggingDatacenter.class.getSimpleName());

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

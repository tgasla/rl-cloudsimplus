package daislab.cspg;

import java.util.List;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.core.Simulation;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class DatacenterWithType extends DatacenterSimple {
    private String type;

    public DatacenterWithType(final Simulation simulation, final List<Host> hostList,
            final VmAllocationPolicy vmAllocationPolicy, final String type) {
        super(simulation, hostList, vmAllocationPolicy);
        this.type = type;
    }
}

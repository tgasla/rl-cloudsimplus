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
    private List<Integer> connectTo;

    public DatacenterWithType(final Simulation simulation, final List<Host> hostList,
            final VmAllocationPolicy vmAllocationPolicy, final String type,
            final List<Integer> connectTo) {
        super(simulation, hostList, vmAllocationPolicy);
        this.type = type;
        this.connectTo = connectTo;
    }
}

package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

public class VmAllocationPolicyRl extends VmAllocationPolicySimple {

    public VmAllocationPolicyRl() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(final Vm vm) {
        final String vmDescription = vm.getDescription();
        final int index = vmDescription.indexOf('-');

        // Desciption does not contain - allocate with VmAllocationPolicySimple logic
        if (index == -1) {
            super.allocateHostForVm(vm);
        }

        final int hostId = Integer.parseInt(vmDescription.substring(index + 1));
        final Host host = getHostList().get(hostId);

        // host with id hostId was not found - allocate with VmAllocationPolicySimple logic
        if (host == host.NULL) {
            super.allocateHostForVm(vm);
        }
        
        // TODO: I need to find a way to give penalty to the agent.
        // For now, do not give any penalty.
        HostSuitability suitability = allocateHostForVm(vm, host);
        if (!suitability.fully()) {
            LOGGER.debug("Action failed because host is not suitable.\n"
            + "Reason: " + suitability.toString());
        }
        return suitability;
    }
}

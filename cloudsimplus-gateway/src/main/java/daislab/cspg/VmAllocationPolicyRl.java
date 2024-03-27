package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

public class VmAllocationPolicyRl extends VmAllocationPolicySimple {

    public VmAllocationPolicyRl() {
        super();
        LOGGER.debug("Calling the VmAllocationPolicySimple constructor");
    }

    @Override
    public HostSuitability allocateHostForVm(final Vm vm) {
        final String vmDescription = vm.getDescription();
        final int index = vmDescription.indexOf('-');
        final HostSuitability suitability;
        final int hostId;
        final Host host;

        if (index == -1) {
            LOGGER.debug("Desciption does not contain the hostId to place the vm."
                    + " The vm will be allocated using the VmAllocationPolicySimple.");
            suitability = super.allocateHostForVm(vm);
            return suitability;
        }

        hostId = Integer.parseInt(vmDescription.substring(index + 1));
        host = getHostList().get(hostId);
        
        // TODO: I need to find a way to give penalty to the agent.
        // For now, do not give any penalty.
        suitability = allocateHostForVm(vm, host);
        if (!suitability.fully()) {
            LOGGER.debug("Action failed because host is not suitable.\n"
            + "Reason: " + suitability.toString());
        }
        return suitability;
    }
}

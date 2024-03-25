package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

public class VmAllocationPolicyRL extends VmAllocationPolicySimple {

    @Override
    public HostSuitability allocateHostForVm(Vm vm) {
        final String vmDescription = vm.getDescription();
        final int hostId = Integer.parseInt(vmDescription.substring(vmDescription.indexOf('-') + 1));
        if (hostId == -1) {
            super.allocateHostForVm(vm);
        }

        final Host host = getHostList().get(hostId);
        
        HostSuitability suitability = allocateHostForVm(vm, host);
        if (!suitability.fully()) {
            LOGGER.debug("Action failed because host is not suitable.\n"
            + "Reason: " + suitability.toString());
        }
        return suitability;
    }
}

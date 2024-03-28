package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

public class VmAllocationPolicyRl extends VmAllocationPolicySimple {

    @Override
    public HostSuitability allocateHostForVm(final Vm vm) {
        final String vmDescription = vm.getDescription();
        final int index = vmDescription.indexOf('-');
        final HostSuitability suitability;
        final int hostId;
        final Host host;

        if (index == -1) {
            LOGGER.debug("Desciption does not contain the hostId to place the vm."
                    + " This is for the initial vms. Placing them using simple policy.");
            suitability = super.allocateHostForVm(vm);
            return suitability;
        }

        hostId = Integer.parseInt(vmDescription.substring(index + 1));
        host = getHostList().get(hostId);
        
        // TODO: This is not needed becuase we check for suitability
        // before telling the broker to try allocate host for this vm
        suitability = allocateHostForVm(vm, host);
        if (!suitability.fully()) {
            LOGGER.debug("This should never be printed as it is already checked.\n"
                    + "Action failed because host is not suitable.\n"
                    + "Reason: " + suitability.toString());
            return suitability;
        }
        LOGGER.debug("New vm creation and allocation with RL policy was successful!");
        return suitability;
    }
}

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
        final long hostId;
        final Host host;

        if (index == -1) {
            LOGGER.debug("Description does not contain the hostId to place the vm."
                + " This is for the initial vms. Placing them using simple policy.");
            suitability = super.allocateHostForVm(vm);
            return suitability;
        }

        hostId = Long.parseLong(vmDescription.substring(index + 1));
        host = getDatacenter().getHostById(hostId);

        suitability = allocateHostForVm(vm, host);
        if (!suitability.fully()) {
            LOGGER.debug("This should never be printed as it is already checked.\n"
                + "Action failed because host is not suitable.\n"
                + "Reason: " + suitability.toString());
            return suitability;
        }

        // set the desctiption to be the vm type
        vm.setDescription(vmDescription.substring(0, index));
        LOGGER.debug("New vm " + vm.getId() + "allocated to host " + hostId);
        return suitability;
    }
}

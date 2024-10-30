package daislab.cspg;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

public class VmAllocationPolicyCustom extends VmAllocationPolicySimple {

    @Override
    public HostSuitability allocateHostForVm(final Vm vm) {
        // The vm description except the vm type contains also the hostId that is supposed to be allocated
        // vm desctiption example: "S-13". We parse the description getting the number after the '-' symbol
        // and after the allocation is done we set the description back to the vm type only (i.e. "S")
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

        // set the desctiption to be the vm type
        vm.setDescription(vmDescription.substring(0, index));
        LOGGER.debug("New vm {} allocated to host {}", vm.getId(), hostId);
        return suitability;
    }
}

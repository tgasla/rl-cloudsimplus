package daislab.cspg;

import org.cloudsimplus.hosts.HostSuitability;

public class VmAllocationPolicyRL extends VmAllocationPolicySimple {
    
    @Override
    public boolean allocateHostForVm(Vm vm) {
        
        List<Vm> vmExecList = broker.getVmExecList();
        Host host = vmExecList.get(id).getHost();
        
        HostSuitability suitability = allocateHostForVm(vm, host);
        if (suitability.fully()) {
            return true;
        }
    
        LOGGER.debug("Action failed because host is not suitable.\n"
        + "Reason: " + suitability.toString());
        return false;
    }
}

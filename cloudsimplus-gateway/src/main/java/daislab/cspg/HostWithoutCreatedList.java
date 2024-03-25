package daislab.cspg;

import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.provisioners.ResourceProvisioner;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.vms.Vm;

import java.util.List;

public class HostWithoutCreatedList extends HostSimple {
    public HostWithoutCreatedList(List<Pe> peList) {
        super(peList);
    }

    public HostWithoutCreatedList(List<Pe> peList, boolean activate) {
        super(peList, activate);
    }

    public HostWithoutCreatedList(ResourceProvisioner ramProvisioner, ResourceProvisioner bwProvisioner, long storage, List<Pe> peList) {
        super(ramProvisioner, bwProvisioner, storage, peList);
    }

    public HostWithoutCreatedList(long ram, long bw, long storage, List<Pe> peList) {
        super(ram, bw, storage, peList);
    }

    public HostWithoutCreatedList(long ram, long bw, long storage, List<Pe> peList, boolean activate) {
        super(ram, bw, storage, peList, activate);
    }

    @Override
    protected void addVmToCreatedList(Vm vm) {
        // do nothing to avoid accumulating vm data
    }
}
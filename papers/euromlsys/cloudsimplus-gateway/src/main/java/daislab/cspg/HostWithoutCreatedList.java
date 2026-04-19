package daislab.cspg;

import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.vms.Vm;
import java.util.List;

public class HostWithoutCreatedList extends HostSimple {
    public HostWithoutCreatedList(final long ram, final long bw, final long storage,
            final List<Pe> peList) {
        super(ram, bw, storage, peList);
    }

    @Override
    protected void addVmToCreatedList(final Vm vm) {
        // Do nothing to avoid accumulating vm data. You cannot do a Host.getVmCreatedList().clear()
        // because it is unmodifiable.
    }
}

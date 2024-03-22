package daislab.csg;

import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.listeners.EventListener;
import org.cloudsimplus.listeners.VmHostEventInfo;
import lombok.NonNull;

public class MyVm extends VmSimple {
    public MyVm(long id, double mipsCapacity, long pesNumber) {
        super(id, mipsCapacity, pesNumber);
    }

    @Override
    public Vm addOnHostAllocationListener(@NonNull EventListener<VmHostEventInfo> listener) {
        super.addOnHostAllocationListener(listener);
        return this;
    }
}

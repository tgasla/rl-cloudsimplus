package daislab.cspg;

import java.util.HashMap;
import java.util.Map;

/*
 * Class to count the number of VMs, so we can keep track of them.
 * We do not allow the creation of a new VM if we have reached the maxVmsPerSize.
 * TODO: check if I can do it using the VmCount functionality of the cloudsimplus.
*/
public class VmCounter {
    private final long maxVmsPerSize;
    private final Map<String, Long> vmCounts = new HashMap<>();

    public VmCounter(final long maxVmsPerSize) {
        this.maxVmsPerSize = maxVmsPerSize;
    }

    public boolean hasCapacity(final String type) {
        return getCurrentOfType(type) < maxVmsPerSize;
    }

    public void recordNewVm(final String type) {
        final Long currentOfType = getCurrentOfType(type);
        this.vmCounts.put(type, currentOfType + 1);
    }

    private Long getCurrentOfType(final String type) {
        return this.vmCounts.getOrDefault(type, 0L);
    }

    public void initializeCapacity(final String type, final long initialVmsCount) {
        this.vmCounts.put(type, initialVmsCount);
    }

    public void recordRemovedVm(final String type) {
        final Long currentOfType = getCurrentOfType(type);
        this.vmCounts.put(type, currentOfType - 1);
    }

    public long getStartedVms(final String type) {
        return getCurrentOfType(type);
    }
}

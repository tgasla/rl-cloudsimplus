package daislab.cspg;

import java.util.HashMap;
import java.util.Map;

/*
 * Class to count the number of VMs per type, so we can keep track them.
 * Otherwise for each step we need to do a parallel stream and count every type, which can be slow.
*/
public class VmCounter {
    private final Map<String, Long> vmCounts = new HashMap<>();

    public void recordNewVm(final String type) {
        final Long currentOfType = getCurrentOfType(type);
        vmCounts.put(type, currentOfType + 1);
    }

    private Long getCurrentOfType(final String type) {
        return vmCounts.getOrDefault(type, 0L);
    }

    public void initializeCapacity(final String type, final long initialVmsCount) {
        vmCounts.put(type, initialVmsCount);
    }

    public void recordRemovedVm(final String type) {
        final Long currentOfType = getCurrentOfType(type);
        vmCounts.put(type, currentOfType - 1);
    }

    public long getStartedVms(final String type) {
        return getCurrentOfType(type);
    }
}

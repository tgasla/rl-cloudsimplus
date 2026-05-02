package daislab.cspg;

/**
 * VM Management action decoder.
 * Translates [action, hostId, vmIndex, vmType] into CloudSim VM create/destroy operations.
 */
public class VmManagementActionDecoder implements IActionDecoder {
    private final WrappedSimulation sim;

    public VmManagementActionDecoder(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public int[] decodeAction(int[] action) {
        return sim.executeCustomAction(action);
    }
}

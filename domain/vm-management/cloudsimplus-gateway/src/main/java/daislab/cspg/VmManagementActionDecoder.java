package daislab.cspg;

/**
 * VM Management action decoder.
 * Decodes RL actions into CloudSim VM management commands.
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
package daislab.cspg;

/**
 * Decodes an RL action (int[]) into CloudSim operations.
 * Implemented by domain-specific inner classes in each WrappedSimulation.
 */
public interface IActionDecoder {
    /**
     * Decode an RL action into CloudSim operations.
     * @param action int[] from the RL agent
     * @return result array — format is domain-specific; [0] is typically validity flag
     */
    int[] decodeAction(int[] action);
}

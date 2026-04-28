package daislab.cspg;

/**
 * Extracts RL state/observation from the simulation.
 * Implemented by domain-specific inner classes in each WrappedSimulation.
 */
public interface IStateExtractor {
    /**
     * @return flat int[] observation for the current simulation state
     */
    int[] extractState();
}

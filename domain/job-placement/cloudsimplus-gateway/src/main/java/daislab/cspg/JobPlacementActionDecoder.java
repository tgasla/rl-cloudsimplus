package daislab.cspg;

/**
 * Job Placement action decoder.
 * Passes through actions directly (one DC index per waiting job).
 */
public class JobPlacementActionDecoder implements IActionDecoder {
    private final WrappedSimulation sim;

    public JobPlacementActionDecoder(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public int[] decodeAction(int[] action) {
        return action;
    }
}
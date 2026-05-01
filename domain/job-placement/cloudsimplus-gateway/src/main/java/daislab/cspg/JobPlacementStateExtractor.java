package daislab.cspg;

/**
 * Job Placement state extractor.
 * Extracts flat per-host [dc_id, dc_type, free_vmpes] observation from the simulation.
 */
public class JobPlacementStateExtractor implements IStateExtractor {
    private final WrappedSimulation sim;

    public JobPlacementStateExtractor(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public int[] extractState() {
        return sim.getInfrastructureObservation();
    }
}
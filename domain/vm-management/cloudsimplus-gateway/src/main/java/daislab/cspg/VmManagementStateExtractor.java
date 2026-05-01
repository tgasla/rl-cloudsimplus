package daislab.cspg;

/**
 * VM Management state extractor.
 * Extracts tree-array infrastructure observation from the simulation.
 */
public class VmManagementStateExtractor implements IStateExtractor {
    private final WrappedSimulation sim;

    public VmManagementStateExtractor(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public int[] extractState() {
        return sim.getInfrastructureObservation();
    }
}
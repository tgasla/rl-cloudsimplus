package daislab.cspg;

/**
 * Job Placement reward calculator.
 * Computes reward based on jobs placed ratio, quality ratio, and deadline violation ratio.
 */
public class JobPlacementRewardCalculator implements IRewardCalculator {
    private final WrappedSimulation sim;

    public JobPlacementRewardCalculator(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public double calculateReward() {
        return sim.getLastReward();
    }
}
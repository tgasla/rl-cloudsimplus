package daislab.cspg;

/**
 * VM Management reward calculator.
 * Computes reward based on job wait time, VM utilization, and action validity.
 */
public class VmManagementRewardCalculator implements IRewardCalculator {
    private final WrappedSimulation sim;

    public VmManagementRewardCalculator(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public double[] calculateReward(boolean isValid) {
        return sim.calculateReward(isValid);
    }
}
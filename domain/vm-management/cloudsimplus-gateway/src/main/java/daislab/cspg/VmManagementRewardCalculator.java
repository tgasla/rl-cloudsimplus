package daislab.cspg;

/**
 * VM Management reward calculator.
 * Computes reward based on job wait time, running VM cores, unutilized cores, and invalid action penalty.
 */
public class VmManagementRewardCalculator implements IRewardCalculator {
    private final WrappedSimulation sim;

    public VmManagementRewardCalculator(WrappedSimulation sim) {
        this.sim = sim;
    }

    @Override
    public double calculateReward() {
        return sim.calculateReward(true)[0];
    }
}

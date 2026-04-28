package daislab.cspg;

/**
 * Computes RL reward from simulation metrics.
 * Implemented by domain-specific inner classes in each WrappedSimulation.
 */
public interface IRewardCalculator {
    /**
     * @param isValid whether the last action was valid
     * @return double[] reward components — [0] is total reward, subsequent entries are breakdown
     */
    double[] calculateReward(boolean isValid);
}

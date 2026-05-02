package daislab.cspg;

/**
 * Computes RL reward from simulation metrics.
 * Implemented by domain-specific inner classes in each WrappedSimulation.
 */
public interface IRewardCalculator {
    /**
     * @return double scalar total reward
     */
    double calculateReward();
}

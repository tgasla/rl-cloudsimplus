package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the object that is returned as a result of each step method call.
 */
public class SimulationStepResult {

    private final double[][] obs;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;
    private final SimulationStepInfo info;

    public SimulationStepResult(final double[][] obs, final double reward, final boolean terminated,
            final boolean truncated, final SimulationStepInfo info) {
        this.obs = obs;
        this.reward = reward;
        this.terminated = terminated;
        this.truncated = truncated;
        this.info = info;
    }

    public double[][] getObs() {
        return obs;
    }

    public double getReward() {
        return reward;
    }

    public boolean isTerminated() {
        return terminated;
    }

    public boolean isTruncated() {
        return truncated;
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" + ", obs=" + Arrays.toString(obs) + ", reward=" + reward
                + ", terminated=" + terminated + ", truncated=" + truncated + ", info="
                + info.toString() + '}';
    }
}

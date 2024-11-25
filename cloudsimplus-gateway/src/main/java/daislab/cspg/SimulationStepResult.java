package daislab.cspg;

/*
 * Class that represents the object that is returned as a result of each step method call.
 */
public class SimulationStepResult {

    private final Observation observation;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;
    private final SimulationStepInfo info;

    public SimulationStepResult(final Observation obs, final double reward,
            final boolean terminated, final boolean truncated, final SimulationStepInfo info) {
        this.observation = obs;
        this.reward = reward;
        this.terminated = terminated;
        this.truncated = truncated;
        this.info = info;
    }

    public Observation getObservation() {
        return observation;
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
        return "SimulationStepResult{" + ", reward=" + reward + ", terminated=" + terminated
                + ", truncated=" + truncated + ", info=" + info.toString() + '}';
    }
}

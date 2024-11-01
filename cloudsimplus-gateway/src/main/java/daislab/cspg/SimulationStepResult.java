package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the object that is returned as a result of each step method call.
 */
public class SimulationStepResult {

    // private final double[][] observation2dArray;
    // private final int[] observationTreeArray;
    private final Observation observation;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;
    private final SimulationStepInfo info;

    public SimulationStepResult(final Observation observation, final double reward,
            final boolean terminated, final boolean truncated, final SimulationStepInfo info) {
        // this.observation2dArray = obs;
        this.observation = observation;
        this.reward = reward;
        this.terminated = terminated;
        this.truncated = truncated;
        this.info = info;
        // this.observationTreeArray = null;
    }

    // public int[] getObservationTreeArray() {
    // return observationTreeArray;
    // }

    // public double[][] getObservation2dArray() {
    // return observation2dArray;
    // }

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

    // @Override
    // public String toString() {
    // return "SimulationStepResult{" + ", observation2dArray="
    // + Arrays.toString(observation2dArray) + ", observationTreeArray="
    // + Arrays.toString(observationTreeArray) + ", reward=" + reward + ", terminated="
    // + terminated + ", truncated=" + truncated + ", info=" + info.toString() + '}';
    // }
}

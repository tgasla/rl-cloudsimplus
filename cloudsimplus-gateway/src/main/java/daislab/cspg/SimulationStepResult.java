package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the object that is returned as a result
 * of each step method call.
*/
public class SimulationStepResult {

    private final double[] obs;
    private final double reward;
    private final boolean done;
    private final SimulationStepInfo info;

    public SimulationStepResult(
        final double[] obs, 
        final double reward,
        final boolean done,
        final SimulationStepInfo info
    ) {
        this.obs = obs;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }

    public double[] getObs() {
        return obs;
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{"
            + ", obs=" + Arrays.toString(obs)
            + ", reward=" + reward
            + ", done=" + done
            + ", info=" + info.toString()
            + '}';
    }
}

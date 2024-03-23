package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the object that is returned as a result
 * of each step method call.
*/
public class SimulationStepResult {

    private final boolean done;
    private final double[] obs;
    private final double reward;

    public SimulationStepResult(boolean done, double[] obs, double reward) {
        this.done = done;
        this.obs = obs;
        this.reward = reward;
    }

    public boolean isDone() {
        return done;
    }

    public double[] getObs() {
        return obs;
    }

    public double getReward() {
        return reward;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{"
                + "done=" + done
                + ", obs=" + Arrays.toString(obs)
                + ", reward=" + reward
                + '}';
    }
}

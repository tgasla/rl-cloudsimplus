package daislab.cspg;

import java.util.Arrays;
import com.google.gson.Gson;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
public class SimulationResetResult {

    private final Gson gson = new Gson();
    private final double[][] obs;
    private final SimulationStepInfo info;

    public SimulationResetResult(final double[][] obs, final SimulationStepInfo info) {
        this.obs = obs;
        this.info = info;
    }

    public double[][] getObs() {
        return obs;
    }

    public String getObsAsJson() {
        return gson.toJson(obs);
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" + ", obs=" + Arrays.toString(obs) + ", info="
                + info.toString() + '}';
    }
}

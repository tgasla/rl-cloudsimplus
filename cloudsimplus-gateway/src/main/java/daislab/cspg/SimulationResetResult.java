package daislab.cspg;

import java.util.Arrays;
import com.google.gson.Gson;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
public class SimulationResetResult {

    // private final Gson gson = new Gson();
    private final Observation observation;
    // private final int[] observationTreeArray;
    // private final double[][] observation2dArray;
    private final SimulationStepInfo info;

    public SimulationResetResult(final Observation observation, final SimulationStepInfo info) {
        this.observation = observation;
        this.info = info;
    }

    public Observation getObservation() {
        return observation;
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    // @Override
    // public String toString() {
    // return "SimulationStepResult{" + ", observation2dArray="
    // + Arrays.toString(observation2dArray) + "observationTreeArray= "
    // + Arrays.toString(observationTreeArray) + ", info=" + info.toString() + '}';
    // }
}

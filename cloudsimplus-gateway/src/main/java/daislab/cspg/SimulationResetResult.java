package daislab.cspg;

import java.util.Arrays;
import com.google.gson.Gson;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
public class SimulationResetResult {

    private final Gson gson = new Gson();
    private final int[] observationTreeArray;
    private final double[][] observation2dArray;
    private final SimulationStepInfo info;

    public SimulationResetResult(final int[] obs, final SimulationStepInfo info) {
        this.observationTreeArray = obs;
        this.info = info;
        this.observation2dArray = null;
    }

    public SimulationResetResult(final double[][] obs, final SimulationStepInfo info) {
        this.observation2dArray = obs;
        this.info = info;
        this.observationTreeArray = null;
    }

    public int[] getObservationTreeArray() {
        return observationTreeArray;
    }

    public double[][] getObservation2dArray() {
        return observation2dArray;
    }

    public String getObservation2dArrayAsJson() {
        return gson.toJson(observation2dArray);
    }

    public String getObservationTreeArrayAsJson() {
        return gson.toJson(observationTreeArray);
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" + ", observation2dArray="
                + Arrays.toString(observation2dArray) + "observationTreeArray= "
                + Arrays.toString(observationTreeArray) + ", info=" + info.toString() + '}';
    }
}

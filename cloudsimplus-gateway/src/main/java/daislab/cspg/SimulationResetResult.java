package daislab.cspg;

import java.util.Arrays;
import com.google.gson.Gson;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
public class SimulationResetResult {

    private final Gson gson = new Gson();
    private final int[] observationTreeArray;
    private final double[][] observationMatrix;
    private final SimulationStepInfo info;

    public SimulationResetResult(final int[] obs, final SimulationStepInfo info) {
        this.observationTreeArray = obs;
        this.info = info;
        this.observationMatrix = null;
    }

    public SimulationResetResult(final double[][] obs, final SimulationStepInfo info) {
        this.observationMatrix = obs;
        this.info = info;
        this.observationTreeArray = null;
    }

    public int[] getObservationTreeArray() {
        return observationTreeArray;
    }

    public double[][] getObservationMatrix() {
        return observationMatrix;
    }

    public String getObservationMatrixAsJson() {
        return gson.toJson(observationMatrix);
    }

    public String getObservationTreeArrayAsJson() {
        return gson.toJson(observationTreeArray);
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" + ", observationMatrix=" + Arrays.toString(observationMatrix)
                + "observationTreeArray= " + Arrays.toString(observationTreeArray) + ", info="
                + info.toString() + '}';
    }
}

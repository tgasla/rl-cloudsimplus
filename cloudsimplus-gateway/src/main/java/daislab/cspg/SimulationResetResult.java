package daislab.cspg;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
public class SimulationResetResult {

    private final Observation observation;
    private final SimulationStepInfo info;

    public SimulationResetResult(final Observation obs, final SimulationStepInfo info) {
        this.observation = obs;
        this.info = info;
    }

    public Observation getObservation() {
        return observation;
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" + "info=" + info.toString() + '}';
    }
}

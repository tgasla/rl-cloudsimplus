package daislab.cspg;

import lombok.Data;

/*
 * Class that represents the object that is returned as a result of each step method call.
 */
@Data
public class SimulationStepResult {
    private final Observation observation;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;
    private final SimulationStepInfo info;
}

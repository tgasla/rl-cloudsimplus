package daislab.cspg;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
@Getter
@AllArgsConstructor
@ToString
public class SimulationResetResult {
    private final Observation observation;
    private final SimulationStepInfo info;
}

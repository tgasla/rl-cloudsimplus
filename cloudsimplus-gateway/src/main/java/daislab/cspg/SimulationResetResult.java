package daislab.cspg;

import lombok.Value;

/*
 * Class that represents the object that is returned as a result of the reset method call.
 */
@Value
public class SimulationResetResult {

    Observation observation;
    SimulationStepInfo info;

    // Lombok generates: all-args constructor, getters, equals, hashCode, toString
}

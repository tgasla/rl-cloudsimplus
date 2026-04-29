package daislab.cspg;

import lombok.Value;

/*
 * Class that represents the object that is returned as a result of each step method call.
 */
@Value
public class SimulationStepResult {

    Observation observation;
    double reward;
    boolean terminated;
    boolean truncated;
    SimulationStepInfo info;

    // Lombok generates: all-args constructor, getters, equals, hashCode, toString
}

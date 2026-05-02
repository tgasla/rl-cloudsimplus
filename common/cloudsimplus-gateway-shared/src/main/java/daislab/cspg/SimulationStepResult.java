package daislab.cspg;

import lombok.Value;

/**
 * Step result returned by simulation.step().
 * Maps to proto StepResult message.
 */
@Value
public class SimulationStepResult {
    Observation observation;
    double reward;
    boolean terminated;
    boolean truncated;
    SimulationStepInfo info;
}
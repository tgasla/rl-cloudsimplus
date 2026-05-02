package daislab.cspg;

import lombok.Value;

/**
 * Reset result returned by simulation.reset().
 * Maps to proto ResetResult message.
 */
@Value
public class SimulationResetResult {
    Observation observation;
    SimulationStepInfo info;
}
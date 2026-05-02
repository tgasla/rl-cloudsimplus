package daislab.cspg;

import lombok.Value;

/**
 * Unified observation for both RL problem types.
 * Maps 1:1 to the unified proto Observation message.
 *
 * Field usage by problem:
 *   VM_MANAGEMENT:  infrastructureObservation (field 1), jobCoresWaitingObservation (field 2)
 *   JOB_PLACEMENT: infrastructureObservation (field 3), jobsWaitingObservation (field 4)
 *
 * Each domain's WrappedSimulation sets only the relevant fields;
 * irrelevant fields are left as empty/zero arrays.
 */
@Value
public class Observation {
    int[] infrastructureObservation;
    int jobCoresWaitingObservation;
    int[] jobsWaitingObservation;
}
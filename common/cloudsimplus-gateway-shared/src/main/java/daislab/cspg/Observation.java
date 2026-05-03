package daislab.cspg;

import lombok.Value;

/**
 * Unified observation for both RL problem types.
 * Maps 1:1 to the unified proto Observation message.
 *
 *   infrastructureObservation: vm-management tree array OR job-placement flat per-host array
 *   secondaryObservation:      vm-management [jobCoresWaiting] (len=1) OR job-placement jobs-waiting array
 */
@Value
public class Observation {
    int[] infrastructureObservation;
    int[] secondaryObservation;
}
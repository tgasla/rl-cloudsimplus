package daislab.cspg;

import lombok.Value;

/**
 * VM Management observation: tree-array infrastructure + job cores waiting.
 */
@Value
public class ObservationVmManagement {
    int[] infrastructureObservation;
    int jobCoresWaitingObservation;
}
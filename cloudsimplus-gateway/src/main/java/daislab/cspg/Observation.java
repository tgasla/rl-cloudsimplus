package daislab.cspg;

import lombok.Data;

@Data
public class Observation {
    private final int[] infrastructureObservation;
    private final int[] jobsWaitingObservation;
}

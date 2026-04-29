package daislab.cspg;

import lombok.Value;

@Value
public class Observation {
    int[] infrastructureObservation;
    int jobCoresWaitingObservation;

    // Lombok generates: all-args constructor, getters, equals, hashCode, toString
}

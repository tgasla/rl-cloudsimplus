package daislab.cspg;

import lombok.Value;

/**
 * Job Placement observation: flat per-host [dc_id, dc_type, free_vmpes] + per-job [cores, location, sensitivity, deadline].
 */
@Value
public class ObservationJobPlacement {
    int[] infrastructureObservation;
    int[] jobsWaitingObservation;
}
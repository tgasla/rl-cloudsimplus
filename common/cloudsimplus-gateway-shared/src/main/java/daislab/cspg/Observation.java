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
    int[] infrastructureObservation;      // field 1: tree array (vm_management) or flat (job_placement)
    int jobCoresWaitingObservation;        // field 2: scalar count (vm_management only)
    int[] flatInfrastructureObservation;  // field 3: per-host [dc_id, dc_type, free_vmpes] (job_placement only)
    int[] jobsWaitingObservation;          // field 4: per-job [cores, location, sens, deadline] (job_placement only)
}
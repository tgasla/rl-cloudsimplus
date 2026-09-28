package daislab.cspg;

import lombok.Value;
import java.util.List;

/**
 * Unified step info for both RL problem types.
 * Maps 1:1 to the unified proto StepInfo message (fields 14-15 are deprecated and unset).
 *
 * Field usage by problem:
 *   VM_MANAGEMENT: fields 1-10 (jobWaitReward, runningVmCoresReward, etc.)
 *   JOB_PLACEMENT: fields 11-13 and 16-24 (jobsWaiting, jobsPlaced, net-SLA reward breakdown)
 *   field 6 (jobWaitTime) is shared by both.
 *
 * Each domain's WrappedSimulation sets only the relevant fields;
 * irrelevant fields are left as 0/false/empty.
 */
@Value
public class SimulationStepInfo {
    // -- VM_MANAGEMENT fields (proto fields 1-10) --
    double jobWaitReward;
    double runningVmCoresReward;
    double unutilizedVmCoresReward;
    double invalidReward;
    boolean valid;
    List<Double> jobWaitTime;
    double unutilizedVmCoreRatio;
    int[] observationTreeArray;
    int hostAffected;
    int coresChanged;

    // -- JOB_PLACEMENT fields (proto fields 11-13, 16-24) --
    int jobsWaiting;
    int jobsPlaced;
    double jobsPlacedRatio;
    double slaValueRealized;
    double slaPenaltyPaid;
    double resourceCost;
    int jobsMet;
    int jobsViolated;
    int jobsExpiredUnplaced;
    double potential;
    double offeredValue;
    double unshapedReward;

    /** Default constructor — all fields zero/empty for reset step info. */
    public SimulationStepInfo() {
        this.jobWaitReward = 0;
        this.runningVmCoresReward = 0;
        this.unutilizedVmCoresReward = 0;
        this.invalidReward = 0;
        this.valid = true;
        this.jobWaitTime = List.of();
        this.unutilizedVmCoreRatio = 0;
        this.observationTreeArray = new int[0];
        this.hostAffected = 0;
        this.coresChanged = 0;
        this.jobsWaiting = 0;
        this.jobsPlaced = 0;
        this.jobsPlacedRatio = 0;
        this.slaValueRealized = 0;
        this.slaPenaltyPaid = 0;
        this.resourceCost = 0;
        this.jobsMet = 0;
        this.jobsViolated = 0;
        this.jobsExpiredUnplaced = 0;
        this.potential = 0;
        this.offeredValue = 0;
        this.unshapedReward = 0;
    }

    /** Full constructor — VM management variant (proto fields 1-10, field 6 shared). */
    public SimulationStepInfo(double[] rewards, List<Double> jobWaitTime,
            double unutilizedVmCoreRatio, int[] observationTreeArray,
            int hostAffected, int coresChanged) {
        this.jobWaitReward = rewards[1];
        this.runningVmCoresReward = rewards[2];
        this.unutilizedVmCoresReward = rewards[3];
        this.invalidReward = rewards[4];
        this.valid = this.invalidReward == 0;
        this.jobWaitTime = jobWaitTime;
        this.unutilizedVmCoreRatio = unutilizedVmCoreRatio;
        this.observationTreeArray = observationTreeArray;
        this.hostAffected = hostAffected;
        this.coresChanged = coresChanged;
        // JOB_PLACEMENT fields unused by vm-management
        this.jobsWaiting = 0;
        this.jobsPlaced = 0;
        this.jobsPlacedRatio = 0;
        this.slaValueRealized = 0;
        this.slaPenaltyPaid = 0;
        this.resourceCost = 0;
        this.jobsMet = 0;
        this.jobsViolated = 0;
        this.jobsExpiredUnplaced = 0;
        this.potential = 0;
        this.offeredValue = 0;
        this.unshapedReward = 0;
    }

    /** Full constructor — JOB_PLACEMENT variant (proto fields 11-13 and 16-24, field 6 shared). */
    public SimulationStepInfo(int jobsWaiting, int jobsPlaced, double jobsPlacedRatio,
            List<Double> jobWaitTime, double slaValueRealized, double slaPenaltyPaid,
            double resourceCost, int jobsMet, int jobsViolated, int jobsExpiredUnplaced,
            double potential, double offeredValue, double unshapedReward) {
        // VM_MANAGEMENT fields unused by job-placement
        this.jobWaitReward = 0;
        this.runningVmCoresReward = 0;
        this.unutilizedVmCoresReward = 0;
        this.invalidReward = 0;
        this.valid = true;
        this.jobWaitTime = jobWaitTime;
        this.unutilizedVmCoreRatio = 0;
        this.observationTreeArray = new int[0];
        this.hostAffected = 0;
        this.coresChanged = 0;
        this.jobsWaiting = jobsWaiting;
        this.jobsPlaced = jobsPlaced;
        this.jobsPlacedRatio = jobsPlacedRatio;
        this.slaValueRealized = slaValueRealized;
        this.slaPenaltyPaid = slaPenaltyPaid;
        this.resourceCost = resourceCost;
        this.jobsMet = jobsMet;
        this.jobsViolated = jobsViolated;
        this.jobsExpiredUnplaced = jobsExpiredUnplaced;
        this.potential = potential;
        this.offeredValue = offeredValue;
        this.unshapedReward = unshapedReward;
    }

    /** Convenience getter for observationTreeArray as list (for proto conversion). */
    public List<Integer> getObservationTreeArrayAsList() {
        java.util.ArrayList<Integer> list = new java.util.ArrayList<>(observationTreeArray.length);
        for (int v : observationTreeArray) list.add(v);
        return list;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo { vm: jobWait=" + jobWaitReward + ", runningVmCores=" + runningVmCoresReward
                + ", invalid=" + invalidReward + " | jp: jobsWaiting=" + jobsWaiting
                + ", jobsPlaced=" + jobsPlaced + ", met=" + jobsMet + ", violated=" + jobsViolated
                + ", unshapedReward=" + unshapedReward + " }";
    }
}
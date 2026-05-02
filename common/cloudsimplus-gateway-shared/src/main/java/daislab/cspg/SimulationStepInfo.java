package daislab.cspg;

import lombok.Value;
import java.util.List;

/**
 * Unified step info for both RL problem types.
 * Maps 1:1 to the unified proto StepInfo message (all 15 fields).
 *
 * Field usage by problem:
 *   VM_MANAGEMENT: fields 1-10 (jobWaitReward, runningVmCoresReward, etc.)
 *   JOB_PLACEMENT: fields 11-15 (jobsWaiting, jobsPlaced, jobsPlacedRatio, etc.)
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

    // -- JOB_PLACEMENT fields (proto fields 11-15) --
    int jobsWaiting;
    int jobsPlaced;
    double jobsPlacedRatio;
    double qualityRatio;
    double deadlineViolationRatio;

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
        this.qualityRatio = 0;
        this.deadlineViolationRatio = 0;
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
        this.qualityRatio = 0;
        this.deadlineViolationRatio = 0;
    }

    /** Full constructor — JOB_PLACEMENT variant (proto fields 11-15, field 6 shared). */
    public SimulationStepInfo(int jobsWaiting, int jobsPlaced,
            double jobsPlacedRatio, double qualityRatio, double deadlineViolationRatio,
            List<Double> jobWaitTime) {
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
        this.qualityRatio = qualityRatio;
        this.deadlineViolationRatio = deadlineViolationRatio;
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
                + ", jobsPlaced=" + jobsPlaced + ", deadlineViol=" + deadlineViolationRatio + " }";
    }
}
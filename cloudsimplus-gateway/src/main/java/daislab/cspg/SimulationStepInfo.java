package daislab.cspg;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

/*
 * Class that represents the info object that is returned as part of the result of each step method
 * call.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class SimulationStepInfo {
    private int jobsWaiting;
    private int jobsPlaced;
    private double jobsPlacedRatio;
    private double qualityRatio;
    private double deadlineViolationRatio;
    private double jobsPlacedReward;
    private double qualityReward;
    private double deadlineViolationReward;
    private List<Double> jobWaitTime;
    // private double jobWaitReward = 0;
    // private double runningVmCoresReward = 0;
    // private double unutilizedVmCoresReward = 0;
    // private double invalidReward = 0;
    // private double unutilizedVmCoreRatio = 0;
    // private boolean valid = true;
    // private int[] observationTreeArray = new int[1];
    // private int hostAffected = 0;
    // private int coresChanged = 0;
    // Metrics for all entities
    // private final double[][] hostMetrics;
    // private final double[][] vmMetrics;
    // private final double[][] jobMetrics;
    // private final String dotString;
}

package daislab.cspg;

import java.util.List;
import java.util.ArrayList;
import com.google.gson.Gson;

/*
 * Class that represents the info object that is returned as part of the result of each step method
 * call.
 */
public class SimulationStepInfo {
    private final Gson gson = new Gson();
    private final double jobWaitReward;
    private final double runningVmCoresReward;
    private final double unutilizedVmCoresReward;
    private final double invalidReward;
    private final boolean valid;

    // Metrics for all entities
    // private final double[][] hostMetrics;
    // private final double[][] vmMetrics;
    // private final double[][] jobMetrics;
    private final List<Double> jobWaitTime;
    private final double unutilizedVmCoreRatio;
    private final int[] observationTreeArray;
    private final int hostAffected;
    private final int coresChanged;
    // private final String dotString;

    public SimulationStepInfo() {
        this.jobWaitReward = 0;
        this.runningVmCoresReward = 0;
        this.unutilizedVmCoresReward = 0;
        this.invalidReward = 0;
        this.unutilizedVmCoreRatio = 0;
        // this.hostMetrics = new double[1][1];
        // this.vmMetrics = new double[1][1];
        // this.jobMetrics = new double[1][1];
        this.jobWaitTime = new ArrayList<>();
        this.valid = true;
        // this.dotString = "";
        this.observationTreeArray = new int[1];
        this.hostAffected = 0;
        this.coresChanged = 0;
    }

    public SimulationStepInfo(final double[] rewards, final List<Double> jobWaitTime,
            final double unutilizedVmCoreRatio, final int[] observationTreeArray,
            final int hostAffected, final int coresChanged) {
        this.jobWaitReward = rewards[1];
        this.runningVmCoresReward = rewards[2];
        this.unutilizedVmCoresReward = rewards[3];
        this.invalidReward = rewards[4];
        // this.hostMetrics = timestepMetrics.get(0);
        // this.vmMetrics = timestepMetrics.get(1);
        // this.jobMetrics = timestepMetrics.get(2);
        this.jobWaitTime = jobWaitTime;
        this.unutilizedVmCoreRatio = unutilizedVmCoreRatio;
        this.valid = this.invalidReward == 0 ? true : false;
        this.observationTreeArray = observationTreeArray;
        this.hostAffected = hostAffected;
        this.coresChanged = coresChanged;
        // this.dotString = dotString;
    }

    public double getJobWaitReward() {
        return jobWaitReward;
    }

    public double getRunningVmCoresReward() {
        return runningVmCoresReward;
    }

    public double getUnutilizedVmCoresReward() {
        return unutilizedVmCoresReward;
    }

    public double getInvalidReward() {
        return invalidReward;
    }

    public boolean isValid() {
        return valid;
    }

    public int getHostAffected() {
        return hostAffected;
    }

    public int getCoresChanged() {
        return coresChanged;
    }

    // public double[][] getHostMetrics() {
    // return hostMetrics;
    // }

    // public double[][] getVmMetrics() {
    // return vmMetrics;
    // }

    // public double[][] getJobMetrics() {
    // return jobMetrics;
    // }

    // public String getHostMetricsAsJson() {
    // return gson.toJson(hostMetrics);
    // }

    // public String getVmMetricsAsJson() {
    // return gson.toJson(vmMetrics);
    // }

    // public String getJobMetricsAsJson() {
    // return gson.toJson(jobMetrics);
    // }

    public String getJobWaitTimeAsJson() {
        return gson.toJson(jobWaitTime);
    }

    public double getUnutilizedVmCoreRatio() {
        return unutilizedVmCoreRatio;
    }

    public String getObservationTreeArrayAsJson() {
        return gson.toJson(observationTreeArray);
    }
    // public String getDotString() {
    // return dotString;
    // }

    @Override
    public String toString() {
        return "SimulationStepInfo { jobWaitReward=" + jobWaitReward + ", runningVmCoresReward="
                + runningVmCoresReward + "unutilizedVmCoresReward=" + unutilizedVmCoresReward
                + "invalidReward=" + invalidReward + '}';
    }
}

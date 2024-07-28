package daislab.cspg;
import java.util.List;
import java.util.ArrayList;
import com.google.gson.Gson;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final Gson gson = new Gson();
    private final double jobWaitReward;
    private final double utilReward;
    private final double invalidReward;
    private final double epJobWaitRewardMean;
    private final double epUtilRewardMean;
    private final int epValidCount;

    // Metrics for all entities
    private final List<long[]> hostMetrics;
    private final List<long[]> vmMetrics;
    private final List<long[]> jobMetrics;
    private final List<Double> jobWaitTime;
    private final double unutilizedActive;
    private final double unutilizedAll;

    public SimulationStepInfo() {
        this.jobWaitReward = 0;
        this.utilReward = 0;
        this.invalidReward = 0;
        this.epJobWaitRewardMean = 0;
        this.epUtilRewardMean = 0;
        this.epValidCount = 0;
        this.hostMetrics = new ArrayList<>();
        this.vmMetrics = new ArrayList<>();
        this.jobMetrics = new ArrayList<>();
        this.jobWaitTime = new ArrayList<>();
        this.unutilizedActive = 0;
        this.unutilizedAll = 0;
    }

    public SimulationStepInfo(
        final double[] rewards,
        final List<Object> episodeRewardStats,
        final List<List<long[]>> timestepMetrics,
        final List<Double> jobWaitTime,
        final double[] unutilizedStats
    ) {
        this.jobWaitReward = rewards[1];
        this.utilReward = rewards[2];
        this.invalidReward = rewards[3];
        this.epJobWaitRewardMean = (double) episodeRewardStats.get(0);
        this.epUtilRewardMean = (double) episodeRewardStats.get(1);
        this.epValidCount = (int) episodeRewardStats.get(2);
        this.hostMetrics = timestepMetrics.get(0);
        this.vmMetrics = timestepMetrics.get(1);
        this.jobMetrics = timestepMetrics.get(2);
        this.jobWaitTime = jobWaitTime;
        this.unutilizedActive = unutilizedStats[0];
        this.unutilizedAll = unutilizedStats[1];
    }

    public double getJobWaitReward() {
        return jobWaitReward;
    }

    public double getUtilReward() {
        return utilReward;
    }

    public double getInvalidReward() {
        return invalidReward;
    }

    public double getEpJobWaitRewardMean() {
        return epJobWaitRewardMean;
    }

    public double getEpUtilRewardMean() {
        return epUtilRewardMean;
    }

    public int getEpValidCount() {
        return epValidCount;
    }

    public List<long[]> getHostMetrics() {
        return hostMetrics;
    }

    public List<long[]> getVmMetrics() {
        return vmMetrics;
    }

    public List<long[]> getJobMetrics() {
        return jobMetrics;
    }

    public String getHostMetricsAsJson() {
        return gson.toJson(hostMetrics);
    }

    public String getVmMetricsAsJson() {
        return gson.toJson(vmMetrics);
    }

    public String getJobMetricsAsJson() {
        return gson.toJson(jobMetrics);
    }

    public String getJobWaitTimeAsJson() {
        return gson.toJson(jobWaitTime);
    }

    public double getUnutilizedActive() {
        return unutilizedActive;
    }

    public double getUnutilizedAll() {
        return unutilizedAll;
    }
 
    @Override
    public String toString() {
        // TODO: I also have to print the Maps.toString() here
        return "SimulationStepInfo{"
            + "jobWaitReward" + jobWaitReward
            + ", utilReward=" + utilReward
            + ", invalidReward=" + invalidReward
            + ", epJobWaitRewardMean" + epJobWaitRewardMean
            + ", epUtilRewardMean=" + epUtilRewardMean
            + ", epValidCount=" + epValidCount
            + '}';
    }
}

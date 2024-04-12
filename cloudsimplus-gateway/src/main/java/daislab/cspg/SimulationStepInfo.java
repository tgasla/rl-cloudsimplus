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
    }

    public SimulationStepInfo(
        final double jobWaitReward,
        final double utilReward,
        final double invalidReward,
        final double epJobWaitRewardMean,
        final double epUtilRewardMean,
        final int epValidCount,
        final List<long[]> hostMetrics,
        final List<long[]> vmMetrics,
        final List<long[]> jobMetrics
    ) {
        this.jobWaitReward = jobWaitReward;
        this.utilReward = utilReward;
        this.invalidReward = invalidReward;
        this.epJobWaitRewardMean = epJobWaitRewardMean;
        this.epUtilRewardMean = epUtilRewardMean;
        this.epValidCount = epValidCount;
        this.hostMetrics = hostMetrics;
        this.vmMetrics = vmMetrics;
        this.jobMetrics = jobMetrics;
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

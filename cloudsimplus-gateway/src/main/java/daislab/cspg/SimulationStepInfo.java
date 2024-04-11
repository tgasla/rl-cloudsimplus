package daislab.cspg;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final double jobWaitReward;
    private final double utilReward;
    private final double invalidReward;
    private final double epJobWaitRewardMean;
    private final double epUtilRewardMean;
    private final int epValidCount;

    public SimulationStepInfo() {
        this.jobWaitReward = 0;
        this.utilReward = 0;
        this.invalidReward = 0;
        this.epJobWaitRewardMean = 0;
        this.epUtilRewardMean = 0;
        this.epValidCount = 0;
    }

    public SimulationStepInfo(
        final double jobWaitReward,
        final double utilReward,
        final double invalidReward,
        final double epJobWaitRewardMean,
        final double epUtilRewardMean,
        final int epValidCount
    ) {
        this.jobWaitReward = jobWaitReward;
        this.utilReward = utilReward;
        this.invalidReward = invalidReward;
        this.epJobWaitRewardMean = epJobWaitRewardMean;
        this.epUtilRewardMean = epUtilRewardMean;
        this.epValidCount = epValidCount;
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

    @Override
    public String toString() {
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

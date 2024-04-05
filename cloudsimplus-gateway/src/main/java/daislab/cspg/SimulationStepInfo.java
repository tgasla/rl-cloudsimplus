package daislab.cspg;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final int validCount;
    private final double meanJobWaitPenalty;
    private final double meanUtilizationPenalty;

    public SimulationStepInfo(
        final int validCount,
        final double meanJobWaitPenalty,
        final double meanUtilizationPenalty
    ) {
        this.validCount = validCount;
        this.meanJobWaitPenalty = meanJobWaitPenalty;
        this.meanUtilizationPenalty = meanUtilizationPenalty;
    }

    public int getValidCount() {
        return validCount;
    }
    
    public double getMeanJobWaitPenalty() {
        return meanJobWaitPenalty;
    }

    public double getMeanUtilizationPenalty() {
        return meanUtilizationPenalty;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo{"
            + "validCount=" + validCount
            + ", meanUtilizationPenalty=" + meanUtilizationPenalty
            + ", meanJobWaitPenalty=" + meanJobWaitPenalty
            + '}';
    }
}

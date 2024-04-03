package daislab.cspg;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final int validCount;
    private final double meanJobWaitPenalty;
    private final double meanCostPenalty;

    public SimulationStepInfo(
            final int validCount,
            final double meanJobWaitPenalty,
            final double meanCostPenalty) {

        this.validCount = validCount;
        this.meanJobWaitPenalty = meanJobWaitPenalty;
        this.meanCostPenalty = meanCostPenalty;
    }

    public int getValidCount() {
        return validCount;
    }
    
    public double getMeanJobWaitPenalty() {
        return meanJobWaitPenalty;
    }

    public double getMeanCostPenalty() {
        return meanCostPenalty;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo{"
                + "validCount=" + validCount
                + ", meanCostPenalty=" + meanCostPenalty
                + ", meanJobWaitPenalty=" + meanJobWaitPenalty
                + '}';
    }
}

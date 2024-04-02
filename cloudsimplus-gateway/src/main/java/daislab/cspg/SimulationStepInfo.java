package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final int validCount;
    private final double cost;

    public SimulationStepInfo(
            final int validCount,
            final double cost) {

        this.validCount = validCount;
        this.cost = cost;
    }

    public int getValidCount() {
        return validCount;
    }

    public double getCost() {
        return cost;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo{"
                + "validCount=" + validCount
                + ", cost=" + cost
                + '}';
    }
}

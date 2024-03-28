package daislab.cspg;

import java.util.Arrays;

/*
 * Class that represents the info object that is returned as part of the
 * result of each step method call.
*/
public class SimulationStepInfo {

    private final boolean successful;
    private final double cost;

    public SimulationStepInfo(final boolean successful,
                                final double cost) {

        this.successful = successful;
        this.cost = cost;
    }

    public boolean isSuccessful() {
        return successful;
    }

    public double getCost() {
        return cost;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo{"
                + "successful=" + successful
                + ", cost=" + cost
                + '}';
    }
}

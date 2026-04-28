package daislab.cspg;

import java.util.Map;

/**
 * Marker interface for simulation settings built by SimulationSettingsBuilder.
 * Both SimulationSettingsVmManagement (common) and SimulationSettingsJobPlacement
 * (common) implement this interface. Concrete type casting is done in
 * each paper's SimulationFactory.
 */
public interface ISimulationSettings {
    boolean isSplitLargeJobs();
    int getMaxJobPes();

    /** Return the params map used to construct this instance (for reconstruction). */
    Map<String, Object> getParams();
}
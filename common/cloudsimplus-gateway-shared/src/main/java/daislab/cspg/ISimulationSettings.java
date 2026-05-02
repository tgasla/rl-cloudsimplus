package daislab.cspg;

import java.util.Map;

/**
 * Common interface for simulation settings used by both vm-management and job-placement domains.
 * Only exposes methods that are actually called by shared components
 * (SimulationFactory, WrappedSimulationBase).
 *
 * Domain-specific accessors remain on the concrete domain SimulationSettings classes.
 */
public interface ISimulationSettings {

    boolean isSplitLargeJobs();
    int getMaxJobPes();
    int getMaxEpisodeLength();

    Map<String, Object> getParams();
}
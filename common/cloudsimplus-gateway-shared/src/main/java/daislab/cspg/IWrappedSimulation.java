package daislab.cspg;

public interface IWrappedSimulation {
    String getIdentifier();
    /**
     * Starts a new episode. A non-empty jobsJson replaces the job list for this episode (same
     * format as createSimulation); an empty one replays the jobs the simulation was created with.
     */
    SimulationResetResult reset(long seed, String jobsJson);

    default SimulationResetResult reset(long seed) {
        return reset(seed, "");
    }
    SimulationStepResult step(int[] action);
    void close();
    String render();
}

package daislab.cspg;

public interface IWrappedSimulation {
    String getIdentifier();
    SimulationResetResult reset(long seed);
    SimulationStepResult step(int[] action);
    void close();
    String render();
}

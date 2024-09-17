package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.Thread;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class MultiSimulationEnvironment {

    private Map<String, WrappedSimulation> simulations =
            Collections.synchronizedMap(new HashMap<>());

    private SimulationFactory simulationFactory = new SimulationFactory();

    private static final Logger LOGGER =
            LoggerFactory.getLogger(MultiSimulationEnvironment.class.getSimpleName());

    public String createSimulation(final Map<String, Object> params, final String jobsAsJson) {
        WrappedSimulation simulation = simulationFactory.create(params, jobsAsJson);
        String identifier = simulation.getIdentifier();

        simulations.put(identifier, simulation);

        return identifier;
    }

    public SimulationResetResult reset(final String simulationIdentifier, final long seed) {
        final WrappedSimulation simulation = getValidSimulation(simulationIdentifier);
        return simulation.reset(seed);
    }

    private void validateIdentifier(final String simulationIdentifier) {
        if (!simulations.containsKey(simulationIdentifier)) {
            throw new IllegalArgumentException(
                    "Simulation with identifier: " + simulationIdentifier + " not found!");
        }
    }

    public void close(final String simulationIdentifier) {
        validateIdentifier(simulationIdentifier);

        final WrappedSimulation simulation = simulations.remove(simulationIdentifier);

        simulation.close();
        // TODO: I should do a memory leak examination.
        LOGGER.debug("Simulation {} terminated. {} simulations still running.",
                simulationIdentifier, simulations.size());
        if (simulations.isEmpty()) {
            LOGGER.debug("All experiments finished running.");
        }
    }

    public String render(final String simulationIdentifier) {
        final WrappedSimulation simulation = getValidSimulation(simulationIdentifier);

        return simulation.render();
    }

    public SimulationStepResult step(final String simulationIdentifier,
            final List<Integer> action) {
        final WrappedSimulation simulation = getValidSimulation(simulationIdentifier);
        final int[] actionPrimitive = action.stream().mapToInt(Integer::intValue).toArray();
        return simulation.step(actionPrimitive);
    }

    public double clock(final String simulationIdentifier) {
        final WrappedSimulation simulation = getValidSimulation(simulationIdentifier);

        return simulation.clock();
    }

    WrappedSimulation getValidSimulation(final String simulationIdentifier) {
        validateIdentifier(simulationIdentifier);

        return simulations.get(simulationIdentifier);
    }

    int getActiveConnections() {
        return simulations.size();
    }
}

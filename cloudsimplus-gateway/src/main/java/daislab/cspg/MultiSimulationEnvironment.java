package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    public String createSimulation(final Map<String, String> maybeParameters) {
        WrappedSimulation simulation = simulationFactory.create(maybeParameters);
        String identifier = simulation.getIdentifier();

        simulations.put(identifier, simulation);

        return identifier;
    }

    public SimulationResetResult reset(final String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);
        return simulation.reset();
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
    }

    public String render(final String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        return simulation.render();
    }

    public SimulationStepResult step(final String simulationIdentifier, final List<Double> action) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);
        final double[] action_double = action.stream()
            .mapToDouble(Double::doubleValue)
            .toArray();
        return simulation.step(action_double);
    }

    public long ping() {
        LOGGER.info("pong");

        return 31415L;
    }

    public void seed(final String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        simulation.seed();
    }

    public double clock(final String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        return simulation.clock();
    }

    public void shutdown() {
        LOGGER.info("Shutting down as per user's request");
        System.exit(0);
    }

    WrappedSimulation retrieveValidSimulation(final String simulationIdentifier) {
        validateIdentifier(simulationIdentifier);

        return simulations.get(simulationIdentifier);
    }
}

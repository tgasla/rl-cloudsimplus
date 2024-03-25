package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class MultiSimulationEnvironment {

    private Map<String, WrappedSimulation> simulations 
        = Collections.synchronizedMap(new HashMap<>());

    private SimulationFactory simulationFactory = new SimulationFactory();

    private static final Logger LOGGER =
            LoggerFactory.getLogger(MultiSimulationEnvironment.class.getSimpleName());

    public String createSimulation(Map<String, String> maybeParameters) {
        WrappedSimulation simulation = simulationFactory.create(maybeParameters);
        String identifier = simulation.getIdentifier();

        simulations.put(identifier, simulation);

        return identifier;
    }

    public double[] reset(String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);
        return simulation.reset();
    }

    private void validateIdentifier(String simulationIdentifier) {
        if (!simulations.containsKey(simulationIdentifier)) {
            throw new IllegalArgumentException(
                "Simulation with identifier: " + simulationIdentifier + " not found!");
        }
    }

    public void close(String simulationIdentifier) {
        validateIdentifier(simulationIdentifier);

        final WrappedSimulation simulation = simulations.remove(simulationIdentifier);

        simulation.close();
    }

    public String render(String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        return simulation.render();
    }

    public SimulationStepResult step(String simulationIdentifier, double[] action) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);
        return simulation.step(action);
    }

    public long ping() {
        LOGGER.info("pong");

        return 31415L;
    }

    public void seed(String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        simulation.seed();
    }

    public double clock(String simulationIdentifier) {
        final WrappedSimulation simulation = retrieveValidSimulation(simulationIdentifier);

        return simulation.clock();
    }

    public void shutdown() {
        LOGGER.info("Shutting down as per user's request");
        System.exit(0);
    }

    WrappedSimulation retrieveValidSimulation(String simulationIdentifier) {
        validateIdentifier(simulationIdentifier);

        return simulations.get(simulationIdentifier);
    }
}

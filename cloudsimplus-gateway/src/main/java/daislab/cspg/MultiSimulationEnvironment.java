package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import py4j.GatewayServer;

import java.util.concurrent.ConcurrentHashMap;
import java.util.List;
import java.util.Map;

public class MultiSimulationEnvironment {

    private final Map<String, WrappedSimulation> simulations = new ConcurrentHashMap<>();

    private final SimulationFactory simulationFactory = new SimulationFactory();

    private GatewayServer gatewayServer;

    private String runMode;

    private int experimentsFinishedCount = 0;
    private int numExperiments = 0;

    private static final Logger LOGGER =
            LoggerFactory.getLogger(MultiSimulationEnvironment.class.getSimpleName());

    public String createSimulation(final Map<String, Object> params, final String jobsAsJson) {
        WrappedSimulation simulation = simulationFactory.create(params, jobsAsJson);
        String identifier = simulation.getIdentifier();
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            System.out.println(entry.getKey() + ":" + entry.getValue().toString());
        }
        runMode = params.get("run_mode").toString();
        numExperiments = (int) params.get("num_experiments");
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
        experimentsFinishedCount++;

        if (runMode.equals("batch")) {
            LOGGER.debug("Simulation {} terminated. {} simulations still running.",
                    simulationIdentifier, simulations.size());
            if (simulations.isEmpty()) {
                LOGGER.debug("All experiments finished running.");
                // Schedule the shutdown after sending the response
                Main.initiateShutdown(gatewayServer);
            }
        } else if (runMode.equals("serial")) {
            LOGGER.debug("Simulation {}/{} terminated", simulationIdentifier, numExperiments);
            if (simulations.isEmpty() && experimentsFinishedCount == numExperiments) {
                LOGGER.debug("All experiments finished running.");
                // Schedule the shutdown after sending the response
                Main.initiateShutdown(gatewayServer);
            }
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

    void setGatewayServer(final GatewayServer gatewayServer) {
        this.gatewayServer = gatewayServer;
    }
}

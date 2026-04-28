package daislab.cspg;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Shared state and logic for gRPC service implementations.
 * Uses composition (not inheritance) to avoid proto dependency lock-in.
 * <p>
 * Both CloudSimGrpcService instances delegate to this class for:
 * <ul>
 *   <li>Simulation map management</li>
 *   <li>Shutdown tracking</li>
 *   <li>Identifier validation</li>
 *   <li>Action array parsing</li>
 * </ul>
 */
public class GrpcServiceDelegate {

    private final ConcurrentHashMap<String, Object> simulations = new ConcurrentHashMap<>();
    private volatile boolean shutdownRequested = false;

    /**
     * Returns the simulation for the given id, or null if not found.
     */
    public Object getSimulation(String simId) {
        return simulations.get(simId);
    }

    /**
     * Stores a simulation under its identifier.
     */
    public void putSimulation(String identifier, Object simulation) {
        simulations.put(identifier, simulation);
    }

    /**
     * Removes and returns the simulation for the given id.
     */
    public Object removeSimulation(String simId) {
        return simulations.remove(simId);
    }

    /**
     * Checks if the simulation map is empty.
     */
    public boolean isSimulationsEmpty() {
        return simulations.isEmpty();
    }

    /**
     * Returns the number of active simulations.
     */
    public int simulationCount() {
        return simulations.size();
    }

    /**
     * Returns true if shutdown has been requested.
     */
    public boolean isShutdownRequested() {
        return shutdownRequested;
    }

    /**
     * Signals that shutdown should occur when all simulations are closed.
     */
    public void requestShutdown() {
        this.shutdownRequested = true;
    }

    /**
     * Validates that the simulation identifier exists in the map.
     * @throws IllegalArgumentException if the identifier is not found
     */
    public void validateIdentifier(String simId) {
        if (!simulations.containsKey(simId)) {
            throw new IllegalArgumentException(
                    "Simulation with identifier: " + simId + " not found!");
        }
    }

    /**
     * Returns the simulation for the given id after validating it exists.
     * @throws IllegalArgumentException if the identifier is not found
     */
    public Object getValidSimulation(String simId) {
        validateIdentifier(simId);
        return simulations.get(simId);
    }

    /**
     * Converts a gRPC ActionList to an int array.
     * @param actionList the list of actions from the request
     * @param simId the simulation id (used for error messages)
     * @return int array of actions
     * @throws IllegalArgumentException if the action list is empty or contains nulls
     */
    public int[] parseActionArray(List<?> actionList, String simId) {
        if (actionList.isEmpty()) {
            throw new IllegalArgumentException("Action list is empty for simId=" + simId);
        }
        int[] actionArray = new int[actionList.size()];
        for (int i = 0; i < actionArray.length; i++) {
            Object element = actionList.get(i);
            if (element == null) {
                throw new IllegalArgumentException(
                        "Action list has null element at index=" + i + " for simId=" + simId);
            }
            actionArray[i] = ((Number) element).intValue();
        }
        return actionArray;
    }

    /**
     * Converts an int array to a List of Integers.
     */
    public static List<Integer> intArrayToList(int[] arr) {
        List<Integer> list = new java.util.ArrayList<>(arr.length);
        for (int val : arr) list.add(val);
        return list;
    }
}

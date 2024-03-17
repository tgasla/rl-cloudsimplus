package daislab.csg;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class VMCountOverflowTest {

    final MultiSimulationEnvironment multiSimulationEnvironment = new MultiSimulationEnvironment();
    final Gson gson = new Gson();

    // Store the original value of the environment variable (if it exists) to restore it later
    private static String datacenterHostsCnt;

    @BeforeAll
    public static void setUp() {
        // Store the original value of DATACENTER_HOSTS_CNT environment variable
        datacenterHostsCnt = System.getProperty("DATACENTER_HOSTS_CNT");

        // Set the new value of DATACENTER_HOSTS_CNT environment variable
        System.setProperty("DATACENTER_HOSTS_CNT", "5");
    }

    @AfterAll
    public static void tearDown() {
        // Clear DATACENTER_HOSTS_CNT to restore it to its original state
        if (datacenterHostsCnt != null) {
            // Restore the original value of DATACENTER_HOSTS_CNT
            System.setProperty("DATACENTER_HOSTS_CNT", datacenterHostsCnt);
            return;
        }
        
        // If the original value didn't exist, clear the property
        System.clearProperty("YOUR_VARIABLE_NAME");
    }

    @Test
    public void testHandleNegativeMi() throws Exception {
        System.out.println("DATACENTER_HOSTS_CNT: " + System.getenv("DATACENTER_HOSTS_CNT"));

        CloudletDescriptor cloudletDescriptor = new CloudletDescriptor(1, 60, -778, 1);

        List<CloudletDescriptor> jobs = Arrays.asList(cloudletDescriptor);
        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));
        parameters.put(SimulationFactory.SIMULATION_SPEEDUP, "60.0");
        parameters.put(SimulationFactory.SPLIT_LARGE_JOBS, "true");
        parameters.put(SimulationFactory.QUEUE_WAIT_PENALTY, "0.00001");
        parameters.put(SimulationFactory.INITIAL_L_VM_COUNT, "1");
        parameters.put(SimulationFactory.INITIAL_M_VM_COUNT, "1");
        parameters.put(SimulationFactory.INITIAL_S_VM_COUNT, "1");

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);

        int i = 0;
        while (i++ < 1000) {
            SimulationStepResult result = multiSimulationEnvironment.step(simulationId, 1);
            System.out.println("Result: " + result);

            if (result.isDone()) {
                break;
            }
        }

        multiSimulationEnvironment.close(simulationId);
        assertTrue(i < 1000, "There should be much less than a 1000 iterations!");
    }
}

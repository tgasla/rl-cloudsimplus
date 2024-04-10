package daislab.cspg;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.File;
import org.apache.commons.io.FileUtils;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class VMCountOverflowTest {

    private static final ArrayList<Double> nopAction = new ArrayList<Double>(List.of(0.0,0.0));

    final MultiSimulationEnvironment multiSimulationEnvironment = new MultiSimulationEnvironment();
    final Gson gson = new Gson();

    @Test
    public void testHandleNegativeMi() throws Exception {
        CloudletDescriptor cloudletDescriptor = new CloudletDescriptor(1, 60, -778, 1);

        List<CloudletDescriptor> jobs = Arrays.asList(cloudletDescriptor);
        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));
        parameters.put(SimulationFactory.SPLIT_LARGE_JOBS, "true");
        parameters.put(SimulationFactory.INITIAL_L_VM_COUNT, "1");
        parameters.put(SimulationFactory.INITIAL_M_VM_COUNT, "1");
        parameters.put(SimulationFactory.INITIAL_S_VM_COUNT, "1");
        parameters.put("SIMULATION_SPEEDUP", "60.0");
        parameters.put("QUEUE_WAIT_PENALTY", "0.00001");
        parameters.put("DATACENTER_HOSTS_CNT", "5");

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);

        int i = 0;
        while (i++ < 1000) {
            SimulationStepResult result = multiSimulationEnvironment.step(simulationId, nopAction);
            System.out.println("Result: " + result);

            if (result.isDone()) {
                break;
            }
        }

        multiSimulationEnvironment.close(simulationId);
        assertTrue(i < 1000, "There should be much less than a 1000 iterations!");
    }
        
    @AfterAll
    public static void deleteJobLogDirectory() {
        // Recursively delete the tempDirectory and its contents
        try {
            FileUtils.deleteDirectory(new File("./logs"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

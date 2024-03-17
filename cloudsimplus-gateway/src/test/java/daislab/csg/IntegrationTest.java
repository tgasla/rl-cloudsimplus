package daislab.csg;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class IntegrationTest {

    final MultiSimulationEnvironment multiSimulationEnvironment = new MultiSimulationEnvironment();
    final Gson gson = new Gson();

    @Test
    public void testPing() {
        final long ping = multiSimulationEnvironment.ping();

        assertEquals(31415L, ping);
    }

    @Test
    public void testSimulationSingleStep() {
        CloudletDescriptor cloudletDescriptor = new CloudletDescriptor(1, 10, 10000, 4);

        List<CloudletDescriptor> jobs = Arrays.asList(cloudletDescriptor);
        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        multiSimulationEnvironment.step(simulationId, 0);
        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testSimulationWithSingleJob() {
        // Job should start after 10 iterations and last for next 10
        // The job below is "large" - it will be split into chunks of 2 cores each
        //
        // In our env we have 3 VMs: 1L, 1M, 1S which are assigned cloudlets
        // in round robin fashion. We want to give each machine a chunk of work
        //
        // This means the job needs to be splitted into 3 equal chunks which we want
        // to last for 10 iterations each. A single core has 10000 MIPS, so we want
        // a the small chunk to have 2*10*10000 MIPS (2 because the smallest machine
        // has 2 cores, doesn't matter for the bigger ones).
        CloudletDescriptor cloudletDescriptor = new CloudletDescriptor(1, 10, (2*10*10000)*3, 2+2+2);

        List<CloudletDescriptor> jobs = Arrays.asList(cloudletDescriptor);
        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, 0);
        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);
            step = multiSimulationEnvironment.step(simulationId, 0);
            stepsExecuted++;
        }
        
        // 21 not 20 because we technically submit the task at 10.1 so it is not able to
        // finish _exactly_ at 20
        assertEquals(21, stepsExecuted);
        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testWithCreatingNewVirtualMachines() {
        // every cloudlet executes for 40 simulation iterations and starts with a delay of 20*i iterations
        List<CloudletDescriptor> jobs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            jobs.add(new CloudletDescriptor(i, 20 * i, 400000, 4));
        }

        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, 0);

        double maxCoreRatio = 0.0;
        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            int action = stepsExecuted == 20 ? 1 : 0;

            step = multiSimulationEnvironment.step(simulationId, action);
            if (step.getObs()[0] > maxCoreRatio) {
                maxCoreRatio = step.getObs()[0];
            }

            System.out.println("Observations: " + Arrays.toString(step.getObs()) + " clock: " + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;
        }

        final WrappedSimulation wrappedSimulation = multiSimulationEnvironment.retrieveValidSimulation(simulationId);
        final SimulationSettings settings = wrappedSimulation.getSimulationSettings();

        final int initialSVmCount = Integer.parseInt(SimulationFactory.INITIAL_VM_COUNT_DEFAULT);
        final int initialMVmCount = initialSVmCount;
        final int initialLVmCount = initialSVmCount;

        final long basicVmPeCount = settings.getBasicVmPeCnt();
        final long totalVmPes =  (initialSVmCount + 1) * basicVmPeCount
                                + initialMVmCount * basicVmPeCount * 2
                                + initialLVmCount * basicVmPeCount * 4;
        
        assertEquals((double) totalVmPes/settings.getDatacenterCores(), maxCoreRatio, 0.000001);

        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testWithDestroyingVMs() {
        List<CloudletDescriptor> jobs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            jobs.add(new CloudletDescriptor(i, 10 * i, 200000, 4));
        }

        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.INITIAL_S_VM_COUNT, "10");
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);

        final WrappedSimulation wrappedSimulation = multiSimulationEnvironment.retrieveValidSimulation(simulationId);
        final SimulationSettings settings = wrappedSimulation.getSimulationSettings();

        final String initialSVmCountStr = parameters.get(SimulationFactory.INITIAL_S_VM_COUNT);
        final int initialSVmCount = Integer.parseInt(initialSVmCountStr);
        final int initialMVmCount = Integer.parseInt(SimulationFactory.INITIAL_VM_COUNT_DEFAULT);
        final int initialLVmCount = initialMVmCount;

        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, 0);

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            if (stepsExecuted == 20) {
                // delete a SMALL VM
                step = multiSimulationEnvironment.step(simulationId, 2);
                final long basicVmPeCount = settings.getBasicVmPeCnt();
                final long totalVmPes = (initialSVmCount - 1) * basicVmPeCount
                                    + initialMVmCount * basicVmPeCount * 2
                                    + initialLVmCount * basicVmPeCount * 4;

                assertEquals((double) totalVmPes/settings.getDatacenterCores(), step.getObs()[0], 0.000001);
            }

            step = multiSimulationEnvironment.step(simulationId, 0);

            System.out.println("Observations: " + Arrays.toString(step.getObs()) + " " + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;
        }
        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testProcessingAllCloudlets() {
        // scenario:
        // 1. we submit jobs at delay 5
        // 2. we have 2S, 1M, 1L VMs
        // 3. we submit enough to overload the system (we have 2+2+4+8 cores, so we submit for 18 cores) for 10 iterations
        //    there should be 2 cloudlets assigned to a VM but not executing
        // 5. we delete the additional S machine at time 10. (at 50% of processing of the accepted jobs)
        // 6. we see what happens to the jobs

        List<CloudletDescriptor> jobs = new ArrayList<>();
        jobs.add(new CloudletDescriptor(1, 5, 100*10000*10, 100));

        Map<String, String> parameters = new HashMap<>();
        parameters.put(SimulationFactory.INITIAL_S_VM_COUNT, "2");
        parameters.put(SimulationFactory.JOBS, gson.toJson(jobs));

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, 0);
        int stepsExecuted = 1;

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            int action = stepsExecuted == 10 ? 2 : 0;
            step = multiSimulationEnvironment.step(simulationId, action);

            System.out.println("Observations: " + Arrays.toString(step.getObs()) + " " + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;

            if (stepsExecuted == 1000) {
                break;
            }
        }
        final WrappedSimulation wrappedSimulation = multiSimulationEnvironment.retrieveValidSimulation(simulationId);
        CloudSimProxy cloudSimProxy = wrappedSimulation.getSimulation();
        cloudSimProxy.printJobStats();
        multiSimulationEnvironment.close(simulationId);

        assertNotEquals(1000, stepsExecuted);
    }
}
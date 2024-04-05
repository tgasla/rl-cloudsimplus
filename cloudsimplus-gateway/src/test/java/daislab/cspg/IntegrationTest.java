package daislab.cspg;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.File;
import org.apache.commons.io.FileUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class IntegrationTest {
        
	private static final int initialMVmCount = 1;
	private static final int initialLVmCount = 1;
	private static final long basicVmPeCount = 2;
	private static final long datacenterHostsCnt = 3000;
	private static final long maxVmsPerSize = datacenterHostsCnt;
	private static final long hostPeMips = 10000;
	private static final int hostPeCnt = 14;
    private static final ArrayList<Double> nopAction = new ArrayList<Double>(List.of(0.0, 0.0));
    
    final MultiSimulationEnvironment multiSimulationEnvironment = new MultiSimulationEnvironment();
    final Gson gson = new Gson();

    private Map<String, String> addParameters(
                        final int initialSVmCount, 
                        final List<CloudletDescriptor> jobs) {

        final Map<String, String> parameters = new HashMap<>();

        parameters.put("INITIAL_S_VM_COUNT", String.valueOf(initialSVmCount));
        parameters.put("INITIAL_M_VM_COUNT", String.valueOf(initialMVmCount));
        parameters.put("INITIAL_L_VM_COUNT", String.valueOf(initialLVmCount));
        parameters.put("DATACENTER_HOSTS_CNT", String.valueOf(datacenterHostsCnt));
        parameters.put("MAX_VMS_PER_SIZE", String.valueOf(maxVmsPerSize));
        parameters.put("HOST_PE_CNT", String.valueOf(hostPeCnt));
        parameters.put("HOST_PE_MIPS", String.valueOf(hostPeMips));
        parameters.put("BASIC_VM_PE_CNT", String.valueOf(basicVmPeCount));
        parameters.put("JOBS", gson.toJson(jobs));
        
        return parameters;
    }

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
        multiSimulationEnvironment.step(simulationId, nopAction);
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
        // a the small chunk to have 2*10*10000 MIPS (2 because the smallest virtual machine
        // has 2 cores, doesn't matter for the bigger ones).
        CloudletDescriptor cloudletDescriptor =
                new CloudletDescriptor(1, 10, 3 * (2 * 10 * 10000), 2 + 2 + 2);

        List<CloudletDescriptor> jobs = Arrays.asList(cloudletDescriptor);

		final int initialSVmCount = 1;

        Map<String, String> parameters = addParameters(initialSVmCount, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);
        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);
            step = multiSimulationEnvironment.step(simulationId, nopAction);
            stepsExecuted++;
        }
        
        // 21 not 20 because we technically submit the task at 10.1 so it is not able to
        // finish _exactly_ at 20
        assertEquals(21, stepsExecuted);
        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testWithCreatingNewVirtualMachines() {
        // every cloudlet executes for 40 simulation iterations
        // and starts with a delay of 20*i iterations
        List<CloudletDescriptor> jobs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            jobs.add(new CloudletDescriptor(i, 20 * i, 400000, 4));
        }

		final int initialSVmCount = 1;

        Map<String, String> parameters = addParameters(initialSVmCount, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);

        double maxCoreRatio = 0.0;
        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            ArrayList<Double> createSVmAction = new ArrayList<Double>(List.of(0.1, 0.0));

            ArrayList<Double> action = stepsExecuted == 20 ? createSVmAction : nopAction;

            System.out.println("action on this step is " + action.get(0) + ", " + action.get(1));

            step = multiSimulationEnvironment.step(simulationId, action);
            if (step.getObs()[0] > maxCoreRatio) {
                maxCoreRatio = step.getObs()[0];
            }

            System.out.println("Observations: " + Arrays.toString(step.getObs())
                    + " clock: " + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;
        }

        final long totalVmPes = 
                (initialSVmCount + 1) * basicVmPeCount
                + initialMVmCount * basicVmPeCount * 2
                + initialLVmCount * basicVmPeCount * 4;

        final long datacenterCores = datacenterHostsCnt * hostPeCnt;

        System.out.println("totalVmPes = " + totalVmPes 
                + " datacenterCores = " + datacenterCores
                + " totalVmPes/datacenterCores = " + (double)totalVmPes/datacenterCores
                + " maxCoreRatio = " + maxCoreRatio);
        
        assertEquals((double) totalVmPes/datacenterCores, maxCoreRatio, 0.000001);

        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testWithDestroyingVMs() {
        List<CloudletDescriptor> jobs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            jobs.add(new CloudletDescriptor(i, 10 * i, 200000, 4));
        }

		final int initialSVmCount = 10;

        Map<String, String> parameters = addParameters(initialSVmCount, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);

        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            if (stepsExecuted == 20) {
                // delete a SMALL VM
                ArrayList<Double> action = new ArrayList<Double>(List.of(-0.1, 0.0));
                step = multiSimulationEnvironment.step(simulationId, action);

                final long totalVmPes =
                        (initialSVmCount - 1) * basicVmPeCount
                        + initialMVmCount * basicVmPeCount * 2
                        + initialLVmCount * basicVmPeCount * 4;

				final long datacenterCores = datacenterHostsCnt * hostPeCnt;

                assertEquals((double) totalVmPes / datacenterCores,
                        step.getObs()[0], 0.000001);
            }

            step = multiSimulationEnvironment.step(simulationId, nopAction);

            System.out.println("Observations: "
                    + Arrays.toString(step.getObs()) + " "
                    + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;
        }
        multiSimulationEnvironment.close(simulationId);
    }

    @Test
    public void testProcessingAllCloudlets() {
        // scenario:
        // 1. we submit jobs at delay 5
        // 2. we have 2S, 1M, 1L VMs
        // 3. we submit enough to overload the system (we have 2+2+4+8 cores,
        //    so we submit for 18 cores) for 10 iterations
        //    there should be 2 cloudlets assigned to a VM but not executing
        // 5. we delete the additional S machine at time 10.
        //    (at 50% of processing of the accepted jobs)
        // 6. we see what happens to the jobs

        List<CloudletDescriptor> jobs = new ArrayList<>();
        jobs.add(new CloudletDescriptor(1, 5, 100*10000*10, 100));

		final int initialSVmCount = 2;

        Map<String, String> parameters = addParameters(initialSVmCount, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);
        int stepsExecuted = 1;

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            ArrayList<Double> removeSVmAction = new ArrayList<Double>(List.of(-0.1, 0.0));

            ArrayList<Double> action = stepsExecuted == 10 ? removeSVmAction : nopAction;
            step = multiSimulationEnvironment.step(simulationId, action);

            System.out.println("Observations: " + Arrays.toString(step.getObs()) + " "
                    + multiSimulationEnvironment.clock(simulationId));
            stepsExecuted++;

            if (stepsExecuted == 1000) {
                break;
            }
        }

        final WrappedSimulation wrappedSimulation =
                multiSimulationEnvironment.retrieveValidSimulation(simulationId);
        CloudSimProxy cloudSimProxy = wrappedSimulation.getSimulation();
        cloudSimProxy.printJobStats();
        multiSimulationEnvironment.close(simulationId);

        assertNotEquals(1000, stepsExecuted);
    }
   
    @AfterAll
    public static void deleteJobLogDirectory() {
        // Recursively delete the tempDirectory and its contents
        try {
            FileUtils.deleteDirectory(new File("./job-logs"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
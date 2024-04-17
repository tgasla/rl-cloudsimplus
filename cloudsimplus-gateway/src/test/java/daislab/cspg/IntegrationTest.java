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
        
	private static final long basicVmPeCount = 2;
	private static final long datacenterHostsCnt = 3000;
	private static final long hostPeMips = 10000;
	private static final int hostPeCnt = 14;
    private static final List<Double> nopAction = new ArrayList<Double>(List.of(0.0, 0.0));
    private static final List<Double> createSVmAction = new ArrayList<Double>(List.of(0.1, 0.0));
    private static final List<Double> removeSVmAction = new ArrayList<Double>(List.of(-0.1, 0.0));
    
    final MultiSimulationEnvironment multiSimulationEnvironment = new MultiSimulationEnvironment();
    final Gson gson = new Gson();

    private Map<String, String> addParameters(
                        final int initialSVmCount,
                        final int initialMVmCount,
                        final int initialLVmCount,
                        final List<CloudletDescriptor> jobs) {

        final Map<String, String> parameters = new HashMap<>();

        parameters.put("INITIAL_S_VM_COUNT", String.valueOf(initialSVmCount));
        parameters.put("INITIAL_M_VM_COUNT", String.valueOf(initialMVmCount));
        parameters.put("INITIAL_L_VM_COUNT", String.valueOf(initialLVmCount));
        parameters.put("DATACENTER_HOSTS_CNT", String.valueOf(datacenterHostsCnt));
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
        parameters.put("JOBS", gson.toJson(jobs));

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

        Map<String, String> parameters = addParameters(1, 1, 1, jobs);

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

        Map<String, String> parameters = addParameters(1, 1, 1, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);

        double maxCoreRatio = 0.0;
        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            List<Double> action = stepsExecuted == 20 ? createSVmAction : nopAction;

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
                2 * basicVmPeCount
                + basicVmPeCount * 2
                + basicVmPeCount * 4;

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

        Map<String, String> parameters = addParameters(10, 1, 1, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);

        int stepsExecuted = 1;
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            if (stepsExecuted == 20) {
                // delete a SMALL VM
                List<Double> action = new ArrayList<Double>(List.of(-0.1, 0.0));
                step = multiSimulationEnvironment.step(simulationId, action);

                final long totalVmPes =
                        (10 - 1) * basicVmPeCount
                        + basicVmPeCount * 2
                        + basicVmPeCount * 4;

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
    public void testNoVmForCloudlet() {
        /*
         * Test to check that if there are not enough resources to host a job at the moment,
         * the job tries to get rescheduled forever.
         */
        List<CloudletDescriptor> jobs = Arrays.asList(new CloudletDescriptor(0, 0, 10, 1));
        Map<String, String> parameters = addParameters(0, 0, 0, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);
        multiSimulationEnvironment.reset(simulationId);

        SimulationStepResult step;
        int stepsExecuted;
        for (stepsExecuted = 1; stepsExecuted < 1000; stepsExecuted++) {
            step = multiSimulationEnvironment.step(simulationId, nopAction);
            System.out.println("Executing step: " + stepsExecuted);
        }
        assertEquals(stepsExecuted, 1000);
    }

    @Test
    public void testNoVmForRescheduledCloudlet() {
        /*
         * Test to check that if there are not enough resources to host a rescheduled job at the moment,
         * the job just waits until there are resources available and reschedules the cloudlet just then.
         * We start the simulation with one S VM.
         * Initially, the cloudlet starts running at 3.1.
         * We remove the VM at step 10.
         * We create a new S VM at step 20. The cloudlet should be rescheduled immediately at the same step.
         * The cloudlet should run for 100000 MI / 10000 MIPS = 10 seconds = 10 timesteps.
         * The simulation should end at 30.1 where we have made 30 steps.
         */
        List<CloudletDescriptor> jobs = Arrays.asList(new CloudletDescriptor(0, 0, 100000, 1));
        Map<String, String> parameters = addParameters(1, 0, 0, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);
        multiSimulationEnvironment.reset(simulationId);

        SimulationStepResult step;
        int stepsExecuted;
        List<Double> action;
        for (stepsExecuted = 0; stepsExecuted < 1000; stepsExecuted++) {
            if (stepsExecuted == 10) {
                action = removeSVmAction;
            }
            else if (stepsExecuted == 21) {
                action = createSVmAction;
            }
            else {
                action = nopAction;
            }
            step = multiSimulationEnvironment.step(simulationId, action);
            System.out.println("Executing step: " + stepsExecuted);
            if (step.isDone()) {
                break;
            }
        }
        assertEquals(stepsExecuted, 31);
    }

    @Test
    public void testProcessingAllCloudlets() {
        // scenario:
        // 1. we submit jobs at delay 5
        // 2. we have 2S, 1M, 1L VMs
        // 3. we submit enough to overload the system (we have 2+2+4+8=16 cores,
        //    so we submit for 18 cores) for 10 iterations
        //    there should be 2 cloudlets assigned to a VM but not executing
        // 5. we delete the additional S machine at time 10.
        //    (at 50% of processing of the accepted jobs)
        // 6. we see what happens to the jobs

        List<CloudletDescriptor> jobs = new ArrayList<>();
        jobs.add(new CloudletDescriptor(1, 5, 100*10000*10, 100));

        Map<String, String> parameters = addParameters(2, 1, 1, jobs);

        final String simulationId = multiSimulationEnvironment.createSimulation(parameters);

        multiSimulationEnvironment.reset(simulationId);
        SimulationStepResult step = multiSimulationEnvironment.step(simulationId, nopAction);
        int stepsExecuted = 1;
        List<Double> action;

        while (!step.isDone()) {
            System.out.println("Executing step: " + stepsExecuted);

            action = stepsExecuted == 10 ? removeSVmAction : nopAction;

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
            FileUtils.deleteDirectory(new File("./logs"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
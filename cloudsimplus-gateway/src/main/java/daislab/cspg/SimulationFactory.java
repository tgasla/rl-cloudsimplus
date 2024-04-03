package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang3.NotImplementedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SimulationFactory {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationFactory.class.getSimpleName());

    private static final Type cloudletDescriptors =
            new TypeToken<List<CloudletDescriptor>>() {}.getType();

    public static final String SPLIT_LARGE_JOBS = "SPLIT_LARGE_JOBS";
    public static final String SPLIT_LARGE_JOBS_DEFAULT = "true";

    public static final String INITIAL_L_VM_COUNT = "INITIAL_L_VM_COUNT";
    public static final String INITIAL_M_VM_COUNT = "INITIAL_M_VM_COUNT";
    public static final String INITIAL_S_VM_COUNT = "INITIAL_S_VM_COUNT";
    public static final String INITIAL_VM_COUNT_DEFAULT = "1";

    public static final String SOURCE_OF_JOBS_PARAMS = "PARAMS";
    public static final String SOURCE_OF_JOBS_FILE = "JOBS_FILE";
    public static final String SOURCE_OF_JOBS_DATABASE = "DB";
    public static final String SOURCE_OF_JOBS = "SOURCE_OF_JOBS";
    public static final String SOURCE_OF_JOBS_DEFAULT = SOURCE_OF_JOBS_PARAMS;
    
    public static final String JOBS = "JOBS";
    public static final String JOBS_EMPTY = "[]";
    public static final String JOBS_DEFAULT = JOBS_EMPTY;

    private static final Gson gson = new Gson();

    private int created = 0;

    public synchronized WrappedSimulation create(final Map<String, String> maybeParameters) {
        String identifier = "Sim" + created;
        created++;

        // get number of initial vms in
        final String initialSVmCountStr =
                maybeParameters.getOrDefault(INITIAL_S_VM_COUNT, INITIAL_VM_COUNT_DEFAULT);
        final int initialSVmCount =
                Integer.parseInt(initialSVmCountStr);
        final String initialMVmCountStr =
                maybeParameters.getOrDefault(INITIAL_M_VM_COUNT, INITIAL_VM_COUNT_DEFAULT);
        final int initialMVmCount =
                Integer.parseInt(initialMVmCountStr);
        final String initialLVmCountStr =
                maybeParameters.getOrDefault(INITIAL_L_VM_COUNT, INITIAL_VM_COUNT_DEFAULT);
        final int initialLVmCount =
                Integer.parseInt(initialLVmCountStr);
        final String sourceOfJobs =
                maybeParameters.getOrDefault(SOURCE_OF_JOBS, SOURCE_OF_JOBS_DEFAULT);
        final String splitLargeJobsStr =
                maybeParameters.getOrDefault(SPLIT_LARGE_JOBS, SPLIT_LARGE_JOBS_DEFAULT);
        final boolean splitLargeJobs =
                Boolean.parseBoolean(splitLargeJobsStr.toLowerCase());

        LOGGER.info("Simulation parameters: ");
        LOGGER.info("-> initialSVmCount: " + initialSVmCount);
        LOGGER.info("-> initialMVmCount: " + initialMVmCount);
        LOGGER.info("-> initialLVmCount: " + initialLVmCount);
        LOGGER.info("-> splitLargeJobs: " + splitLargeJobs);

        final SimulationSettings settings = new SimulationSettings(maybeParameters);
        LOGGER.info("Simulation settings dump");
        LOGGER.info(settings.toString());

        final List<CloudletDescriptor> jobs;

        switch (sourceOfJobs) {
            case SOURCE_OF_JOBS_DATABASE:
                jobs = loadJobsFromDatabase(maybeParameters);
                break;
            case SOURCE_OF_JOBS_FILE:
                jobs = loadJobsFromFile(maybeParameters);
                break;
            case SOURCE_OF_JOBS_PARAMS:
                // fall-through
            default:
                jobs = loadJobsFromParams(maybeParameters, settings.getSimulationSpeedup());
        }

        final List<CloudletDescriptor> splittedJobs;
        if (splitLargeJobs) {
            LOGGER.info("Splitting large jobs");
            splittedJobs = splitLargeJobs(jobs, settings);
        } else {
            LOGGER.info("Not applying the splitting algorithm - using raw jobs");
            splittedJobs = jobs;
        }

        return new WrappedSimulation(
                settings,
                identifier,
                new HashMap<String, Integer>() {{
                    this.put(CloudSimProxy.SMALL, initialSVmCount);
                    this.put(CloudSimProxy.MEDIUM, initialMVmCount);
                    this.put(CloudSimProxy.LARGE, initialLVmCount);
                }},
                splittedJobs);
    }

    private List<CloudletDescriptor> splitLargeJobs(
            final List<CloudletDescriptor> jobs, 
            final SimulationSettings settings) {
        // There is a bug in CloudletSchedulerAbstract:cloudletExecutedInstructionsForTimeSpan
        // The bug is that we take into the account only a single core of a PE
        // when we calculate the available MIPS per cloudlet. So we artificially limit
        // the parallelism to 1 core per task even if there is nothing else to process.
        // (in practice the cloudlet blocks two cores, but uses only one! so the
        // processing time is 2x of what is expected).
        //final int hostPeCnt = (int) settings.getBasicVmPeCnt();
        final int hostPeCnt = 1;

        int splittedId = jobs.size() * 10;

        List<CloudletDescriptor> splitted = new ArrayList<>();
        for(CloudletDescriptor cloudletDescriptor : jobs) {
            int numberOfCores = cloudletDescriptor.getNumberOfCores();
            if (numberOfCores <= hostPeCnt) {
                splitted.add(cloudletDescriptor);
            } else {
                final long miPerCore =
                        (cloudletDescriptor.getMi() / cloudletDescriptor.getNumberOfCores());
                while(numberOfCores > 0) {
                    final int fractionOfCores =
                            hostPeCnt < numberOfCores ? hostPeCnt : numberOfCores;
                    final long totalMips = miPerCore * fractionOfCores;
                    final long totalMipsNotZero = totalMips == 0 ? 1 : totalMips;
                    CloudletDescriptor splittedDescriptor = new CloudletDescriptor(
                            splittedId,
                            cloudletDescriptor.getSubmissionDelay(),
                            totalMipsNotZero,
                            fractionOfCores);
                    splitted.add(splittedDescriptor);
                    splittedId++;
                    numberOfCores -= hostPeCnt;
                }
            }
        }

        LOGGER.info("Splitted: " + jobs.size() + " into " + splitted.size());

        return splitted;
    }

    private List<CloudletDescriptor> loadJobsFromParams(
            final Map<String, String> maybeParameters, 
            final double simulationSpeedup) {

        List<CloudletDescriptor> retVal = new ArrayList<>();
        final String jobsAsJson = maybeParameters.getOrDefault(JOBS, JOBS_DEFAULT);
        LOGGER.info(jobsAsJson);
        final List<CloudletDescriptor> deserialized =
                gson.fromJson(jobsAsJson, cloudletDescriptors);

        for (CloudletDescriptor cloudletDescriptor : deserialized) {
            retVal.add(speedup(cloudletDescriptor, simulationSpeedup));
        }

        LOGGER.info("Deserialized " + retVal.size() + " jobs");

        return retVal;
    }

    private CloudletDescriptor speedup(
            final CloudletDescriptor cloudletDescriptor, 
            final double simulationSpeedup) {
                
        final long cloudletDescriptorMi = cloudletDescriptor.getMi();
        final long nonNegativeMi = cloudletDescriptorMi < 1 ? 1 : cloudletDescriptorMi;
        final long speededUpMi = (long) (nonNegativeMi / simulationSpeedup);
        final long newMi = speededUpMi == 0 ? 1 : speededUpMi;
        final long submissionDelayReal = cloudletDescriptor
                .getSubmissionDelay() < 0 ? 0L : cloudletDescriptor.getSubmissionDelay();

        final long submissionDelay = (long) (submissionDelayReal / simulationSpeedup);
        final int numberOfCores = cloudletDescriptor.
                getNumberOfCores() < 0 ? 1 : cloudletDescriptor.getNumberOfCores();

        return new CloudletDescriptor(
                cloudletDescriptor.getJobId(),
                submissionDelay,
                newMi,
                numberOfCores
        );
    }

    private List<CloudletDescriptor> loadJobsFromDatabase(
            final Map<String, String> maybeParameters) {
        throw new NotImplementedException("Feature not implemented yet!");
    }

    private List<CloudletDescriptor> loadJobsFromFile(
                final Map<String, String> maybeParameters) {
        throw new NotImplementedException("Feature not implemented yet!");
    }

}

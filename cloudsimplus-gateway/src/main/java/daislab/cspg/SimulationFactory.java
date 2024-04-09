package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang3.NotImplementedException;
import org.cloudsimplus.cloudlets.Cloudlet;
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

    public static final String MAX_PES_PER_JOB = "MAX_PES_PER_JOB";
    public static final String MAX_PES_PER_JOB_DEFAULT = "1";

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
        final String maxPesPerJobStr =
            maybeParameters.getOrDefault(MAX_PES_PER_JOB, MAX_PES_PER_JOB_DEFAULT);
        final int maxPesPerJob =
            Integer.parseInt(maxPesPerJobStr);

        LOGGER.info("Simulation parameters: ");
        LOGGER.info("-> initialSVmCount: " + initialSVmCount);
        LOGGER.info("-> initialMVmCount: " + initialMVmCount);
        LOGGER.info("-> initialLVmCount: " + initialLVmCount);
        LOGGER.info("-> splitLargeJobs: " + splitLargeJobs);
        LOGGER.info("-> maxPesPerJob: " + maxPesPerJob);

        final SimulationSettings settings = new SimulationSettings(maybeParameters);
        LOGGER.info("Simulation settings dump");
        LOGGER.info(settings.toString());

        List<CloudletDescriptor> jobs;

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

        if (splitLargeJobs) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, maxPesPerJob, settings);
        }

        return new WrappedSimulation(
			settings,
			identifier,
			new HashMap<String, Integer>() {{
				this.put(CloudSimProxy.SMALL, initialSVmCount);
				this.put(CloudSimProxy.MEDIUM, initialMVmCount);
				this.put(CloudSimProxy.LARGE, initialLVmCount);
			}},
			jobs);
    }

	private List<CloudletDescriptor> splitLargeJobs(
        final List<CloudletDescriptor> jobs,
        final int maxPesPerJob,
        final SimulationSettings settings
	) {

        List<CloudletDescriptor> splitted = new ArrayList<>();
        int splittedId = 0;

		for (CloudletDescriptor cloudletDescriptor : jobs) {
			int jobPesNumber = cloudletDescriptor.getNumberOfCores();
			int splitCount = Math.max((jobPesNumber + maxPesPerJob - 1) / maxPesPerJob, 1);
			int normalSplitPesNumber = jobPesNumber / splitCount;
			long totalMi = cloudletDescriptor.getMi();
			long normalSplitMi = totalMi / splitCount;

			// Distribute the MI and PEs for each split part
			for (int i = 0; i < splitCount; i++) {
				// Last split might have different MI due to remainder
				long miForThisSplit = (i < splitCount - 1) 
					? normalSplitMi 
					: totalMi - (normalSplitMi * (splitCount - 1));
				int pesForThisSplit = (i < splitCount - 1) 
					? normalSplitPesNumber 
					: jobPesNumber - (normalSplitPesNumber * (splitCount - 1));
				CloudletDescriptor splittedDescriptor = new CloudletDescriptor(
					splittedId++, 
					cloudletDescriptor.getSubmissionDelay(),
					// we divide by the pesNumber because cloudSimPlus do not parallelize jobs :)
					// so we artificially do it here.
					miForThisSplit / pesForThisSplit, 
					pesForThisSplit
				);
				
				splitted.add(splittedDescriptor);
			}
		}

		LOGGER.info("Splitted: " + jobs.size() + " into " + splitted.size());
		return splitted;
	}

    private List<CloudletDescriptor> loadJobsFromParams(
        final Map<String, String> maybeParameters, 
        final double simulationSpeedup
    ) {

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
        final double simulationSpeedup
    ) {
    
        final long cloudletMi = cloudletDescriptor.getMi();
        final long newMi = Math.max(cloudletMi, 1);
        // if you uncomment the line below, then simulationSpeedup controls two things:
        // 1. how fast are jobs coming and
        // 2. how long do jobs run
        // Hoever, it is better to control only 1. and let 2. be controlled by the
        // MIPS number in utils.py
        // final long newMi = Math.max((long) (cloudletMi / simulationSpeedup), 1);
        final long cloudletDelay = cloudletDescriptor.getSubmissionDelay();
        final long newDelay = Math.max((long) (cloudletDelay / simulationSpeedup), 0);
        final int pesNumber = Math.max(cloudletDescriptor.getNumberOfCores(), 1);

        return new CloudletDescriptor(
			cloudletDescriptor.getJobId(),
			newDelay,
			newMi,
			pesNumber
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

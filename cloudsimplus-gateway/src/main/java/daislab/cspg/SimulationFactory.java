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

    private static final Gson gson = new Gson();

    private int created = 0;

    public synchronized WrappedSimulation create(final String jobsAsJson) {
        String identifier = "Sim" + created;
        created++;

        final SimulationSettings settings = new SimulationSettings();
        LOGGER.info("Simulation settings dump");
        LOGGER.info(settings.toString());

        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);

        if (settings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, settings.getMaxJobPes());
        }

        return new WrappedSimulation(identifier, settings, jobs);
    }

    private List<CloudletDescriptor> splitLargeJobs(final List<CloudletDescriptor> jobs,
            final int maxJobPes) {
        List<CloudletDescriptor> splitted = new ArrayList<>();
        int splittedId = 0;

        for (CloudletDescriptor cloudletDescriptor : jobs) {
            int jobPesNumber = cloudletDescriptor.getNumberOfCores();
            int splitCount = Math.max(1, (jobPesNumber + maxJobPes - 1) / maxJobPes);
            int normalSplitPesNumber = jobPesNumber / splitCount;
            long totalMi = cloudletDescriptor.getMi();
            // long normalSplitMi = totalMi / splitCount;

            // Distribute the PEs for each split part
            for (int i = 0; i < splitCount; i++) {
                // Last split might have different MI and PES due to remainder
                // long miForThisSplit = (i < splitCount - 1)
                // ? normalSplitMi
                // : totalMi - (normalSplitMi * (splitCount - 1));
                // uncomment the 4 previous lines if you want the splitting algorithm to split
                // the MIs of jobs also
                long miForThisSplit = totalMi;
                int pesForThisSplit = (i < splitCount - 1) ? normalSplitPesNumber
                        : jobPesNumber - (normalSplitPesNumber * (splitCount - 1));
                CloudletDescriptor splittedDescriptor = new CloudletDescriptor(splittedId++,
                        cloudletDescriptor.getSubmissionDelay(),
                        // we divide by the pesNumber because cloudSimPlus do not parallelize jobs
                        // :)
                        // so we artificially do it here.
                        miForThisSplit / pesForThisSplit, pesForThisSplit);

                splitted.add(ensureMinValues(splittedDescriptor));
            }
        }

        LOGGER.info("Splitted: " + jobs.size() + " into " + splitted.size());
        return splitted;
    }

    private List<CloudletDescriptor> loadJobsFromJson(final String jobsAsJson) {
        List<CloudletDescriptor> jobList = new ArrayList<>();
        LOGGER.info(jobsAsJson);
        final List<CloudletDescriptor> deserialized =
                gson.fromJson(jobsAsJson, cloudletDescriptors);

        for (CloudletDescriptor cloudletDescriptor : deserialized) {
            jobList.add(ensureMinValues(cloudletDescriptor));
        }

        LOGGER.info("Deserialized " + jobList.size() + " jobs");

        return jobList;
    }

    private CloudletDescriptor ensureMinValues(final CloudletDescriptor cloudletDescriptor) {
        final long mi = Math.max(1, cloudletDescriptor.getMi());
        final long cloudletDelay = Math.max(0, cloudletDescriptor.getSubmissionDelay());
        final int pesNumber = Math.max(1, cloudletDescriptor.getNumberOfCores());

        return new CloudletDescriptor(cloudletDescriptor.getJobId(), cloudletDelay, mi, pesNumber);
    }
}

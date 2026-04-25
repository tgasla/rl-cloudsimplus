package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class SimulationFactory {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationFactory.class.getSimpleName());

    private static final Type cloudletDescriptorsType =
            new TypeToken<List<CloudletDescriptor>>() {}.getType();

    private static final Gson gson = new Gson();

    private int simulationsRunning = 0;

    public synchronized WrappedSimulation create(final String paramsAsJson,
            final String jobsAsJson) {
        String identifier = "Sim" + simulationsRunning++;

        // Parse JSON params to Map, then use SimulationSettingsBuilder for unified
        // problem-type-aware settings with safe defaults for inactive fields
        com.google.gson.JsonParser parser = new com.google.gson.JsonParser();
        com.google.gson.JsonElement elem = parser.parse(paramsAsJson);
        java.util.Map<String, Object> paramsMap = new java.util.HashMap<>();
        if (elem.isJsonObject()) {
            for (java.util.Map.Entry<String, com.google.gson.JsonElement> e : elem.getAsJsonObject().entrySet()) {
                Object value = e.getValue().isJsonNull() ? null : e.getValue().getAsNumber();
                paramsMap.put(e.getKey(), value);
            }
        }

        // Use SimulationSettingsBuilder for unified problem-type-aware settings
        SimulationSettings settings =
                (SimulationSettings) SimulationSettingsBuilder.build(paramsMap);

        LOGGER.info("Simulation settings dump:\n{}", settings);

        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);

        if (settings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, settings.getMaxJobPes());
        }

        return new WrappedSimulation(identifier, settings, jobs);
    }

    private List<CloudletDescriptor> loadJobsFromJson(final String jobsAsJson) {
        List<CloudletDescriptor> jobList = new ArrayList<>();
        LOGGER.info(jobsAsJson);
        final List<CloudletDescriptor> deserialized =
                gson.fromJson(jobsAsJson, cloudletDescriptorsType);
        for (CloudletDescriptor cloudletDescriptor : deserialized) {
            jobList.add(cloudletDescriptor);
        }
        LOGGER.info("Deserialized {} jobs", jobList.size());
        return jobList;
    }

    private List<CloudletDescriptor> splitLargeJobs(final List<CloudletDescriptor> jobs,
            final int maxJobPes) {
        List<CloudletDescriptor> splitted = new ArrayList<>();
        int splittedId = 0;
        for (CloudletDescriptor cloudletDescriptor : jobs) {
            int jobPesNumber = cloudletDescriptor.getCores();
            int splitCount = Math.max(1, (jobPesNumber + maxJobPes - 1) / maxJobPes);
            int normalSplitPesNumber = jobPesNumber / splitCount;
            long totalMi = cloudletDescriptor.getMi();

            for (int i = 0; i < splitCount; i++) {
                long miForThisSplit = totalMi;
                int pesForThisSplit = (i < splitCount - 1) ? normalSplitPesNumber
                        : jobPesNumber - (normalSplitPesNumber * (splitCount - 1));
                CloudletDescriptor splittedDescriptor = new CloudletDescriptor(splittedId++,
                        cloudletDescriptor.getSubmissionDelay(),
                        miForThisSplit / pesForThisSplit, pesForThisSplit);

                splitted.add(splittedDescriptor);
            }
        }

        LOGGER.info("Splitted: {} into {}", jobs.size(), splitted.size());
        return splitted;
    }
}
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
                com.google.gson.JsonElement val = e.getValue();
                Object value;
                if (val.isJsonNull()) {
                    value = null;
                } else if (val.isJsonPrimitive()) {
                    com.google.gson.JsonPrimitive prim = val.getAsJsonPrimitive();
                    if (prim.isNumber()) {
                        value = prim.getAsNumber();
                    } else if (prim.isBoolean()) {
                        value = prim.getAsBoolean();
                    } else {
                        value = prim.getAsString();
                    }
                } else {
                    value = val.toString();
                }
                paramsMap.put(e.getKey(), value);
            }
        }

        // Use SimulationSettingsBuilder for unified problem-type-aware settings
        // and problem-type detection. Builder returns ISimulationSettings (vm or job placement).
        ISimulationSettings iSettings = SimulationSettingsBuilder.build(paramsMap);

        LOGGER.info("Simulation settings dump:\n{}", iSettings);

        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);

        if (iSettings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, iSettings.getMaxJobPes());
        }

        // Reconstruct paper-specific SimulationSettings from paramsMap
        // Use paper-specific SimulationSettings constructor with the builder's filled params map
        // (which has safe defaults for all fields pre-injected by SimulationSettingsBuilder)
        SimulationSettings settings = new SimulationSettings(iSettings.getParams());
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
package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.JsonParser;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class SimulationFactory {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationFactory.class.getSimpleName());

    private static final Gson gson = new Gson();
    private static final Type cloudletDescriptorsType =
            new TypeToken<List<CloudletDescriptor>>() {}.getType();

    private int simulationsRunning = 0;

    public synchronized WrappedSimulation create(final String paramsAsJson,
            final String jobsAsJson) {
        String identifier = "Sim" + simulationsRunning++;

        // Parse JSON params to Map, then use SimulationSettingsBuilder for unified
        // problem-type-aware settings with safe defaults for inactive fields
        JsonObject paramsObj = JsonParser.parseString(paramsAsJson).getAsJsonObject();
        java.util.Map<String, Object> paramsMap = new java.util.HashMap<>();
        for (java.util.Map.Entry<String, JsonElement> e : paramsObj.entrySet()) {
            JsonElement val = e.getValue();
            if (val.isJsonNull()) {
                paramsMap.put(e.getKey(), null);
            } else if (val.isJsonArray()) {
                Type listType = new TypeToken<List<java.util.Map<String, Object>>>() {}.getType();
                paramsMap.put(e.getKey(), gson.fromJson(val, listType));
            } else if (val.isJsonPrimitive()) {
                try { paramsMap.put(e.getKey(), val.getAsNumber()); }
                catch (Exception ex) { paramsMap.put(e.getKey(), val.getAsString()); }
            } else {
                paramsMap.put(e.getKey(), val.toString());
            }
        }

        // Use SimulationSettingsBuilder for unified problem-type-aware settings
        ISimulationSettings iSettings = SimulationSettingsBuilder.build(paramsMap);

        LOGGER.info("Simulation settings dump:\n{}", iSettings);

        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);

        if (iSettings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, iSettings.getMaxJobPes());
        }

        // Reconstruct paper-specific SimulationSettings from builder's filled params
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
                        miForThisSplit / pesForThisSplit, pesForThisSplit,
                        cloudletDescriptor.getLocation(),
                        cloudletDescriptor.getDelaySensitivity(),
                        cloudletDescriptor.getDeadline());

                splitted.add(splittedDescriptor);
            }
        }

        LOGGER.info("Splitted: {} into {}", jobs.size(), splitted.size());
        return splitted;
    }
}

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
            new TypeToken<List<CloudletDescriptorWithLocation>>() {}.getType();

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

        // Directly construct domain-specific SimulationSettings from params map.
        ISimulationSettings iSettings = new SimulationSettings(paramsMap);

        LOGGER.info("Simulation settings dump:\n{}", iSettings);

        List<CloudletDescriptorWithLocation> jobs = loadJobsFromJson(jobsAsJson);

        if (iSettings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, iSettings.getMaxJobPes());
        }

        // Use domain-specific settings directly
        return new WrappedSimulation(identifier, iSettings, new ArrayList<>(jobs));
    }

    private List<CloudletDescriptorWithLocation> loadJobsFromJson(final String jobsAsJson) {
        LOGGER.info(jobsAsJson);
        final List<CloudletDescriptorWithLocation> deserialized =
                gson.fromJson(jobsAsJson, cloudletDescriptorsType);
        LOGGER.info("Deserialized {} jobs", deserialized.size());
        return deserialized;
    }

    private List<CloudletDescriptorWithLocation> splitLargeJobs(
            final List<CloudletDescriptorWithLocation> jobs, final int maxJobPes) {
        List<CloudletDescriptorWithLocation> splitted = new ArrayList<>();
        int splittedId = 0;
        for (CloudletDescriptorWithLocation cdl : jobs) {
            int jobPesNumber = cdl.getCores();
            int splitCount = Math.max(1, (jobPesNumber + maxJobPes - 1) / maxJobPes);
            int normalSplitPesNumber = jobPesNumber / splitCount;
            long totalMi = cdl.getMi();

            for (int i = 0; i < splitCount; i++) {
                long miForThisSplit = totalMi;
                int pesForThisSplit = (i < splitCount - 1) ? normalSplitPesNumber
                        : jobPesNumber - (normalSplitPesNumber * (splitCount - 1));
                splitted.add(new CloudletDescriptorWithLocation(splittedId++,
                        cdl.getSubmissionDelay(),
                        miForThisSplit / pesForThisSplit, pesForThisSplit,
                        cdl.getLocation(), cdl.getDelaySensitivity(), cdl.getDeadline()));
            }
        }

        LOGGER.info("Splitted: {} into {}", jobs.size(), splitted.size());
        return splitted;
    }
}

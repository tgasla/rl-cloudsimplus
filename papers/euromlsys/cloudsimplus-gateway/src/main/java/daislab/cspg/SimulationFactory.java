package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.google.gson.JsonParser;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

public class SimulationFactory {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationFactory.class.getSimpleName());

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
            } else if (val.isJsonPrimitive()) {
                // Try to get as Number first for numeric params, otherwise as String
                try { paramsMap.put(e.getKey(), val.getAsNumber()); }
                catch (Exception ex) { paramsMap.put(e.getKey(), val.getAsString()); }
            } else {
                paramsMap.put(e.getKey(), val.toString());
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

    private List<CloudletDescriptor> splitLargeJobs(final List<CloudletDescriptor> jobs,
            final int maxJobPes) {
        List<CloudletDescriptor> splitted = new ArrayList<>();
        int splittedId = 0;
        for (CloudletDescriptor cloudletDescriptor : jobs) {
            int jobPesNumber = cloudletDescriptor.getCores();
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
                        // :) so we artificially do it here.
                        miForThisSplit / pesForThisSplit, pesForThisSplit,
                        cloudletDescriptor.getLocation(), cloudletDescriptor.getDelaySensitivity(),
                        cloudletDescriptor.getDeadline());

                splitted.add(splittedDescriptor);
            }
        }

        LOGGER.info("Splitted: {} into {}", jobs.size(), splitted.size());
        return splitted;
    }

    private List<CloudletDescriptor> loadJobsFromJson(final String jobsAsJson) {
        List<CloudletDescriptor> jobList = new ArrayList<>();
        LOGGER.info(jobsAsJson);
        com.google.gson.JsonArray arr = JsonParser.parseString(jobsAsJson).getAsJsonArray();
        for (int i = 0; i < arr.size(); i++) {
            com.google.gson.JsonObject obj = arr.get(i).getAsJsonObject();
            jobList.add(new CloudletDescriptor(
                    obj.get("job_id").getAsInt(),
                    obj.get("submission_delay").getAsLong(),
                    obj.get("mi").getAsLong(),
                    obj.get("cores").getAsInt(),
                    obj.has("location") ? obj.get("location").getAsInt() : 0,
                    obj.has("delay_sensitivity") ? obj.get("delay_sensitivity").getAsInt() : 0,
                    obj.has("deadline") ? obj.get("deadline").getAsInt() : Integer.MAX_VALUE));
        }
        LOGGER.info("Deserialized {} jobs", jobList.size());
        return jobList;
    }
}

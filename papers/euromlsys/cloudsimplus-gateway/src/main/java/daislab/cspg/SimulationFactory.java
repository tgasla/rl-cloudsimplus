package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.google.gson.JsonParser;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

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
                // Parse datacenter arrays properly
                if ("datacenters".equals(e.getKey())) {
                    Type listType = new TypeToken<List<Map<String, Object>>>() {}.getType();
                    paramsMap.put(e.getKey(), gson.fromJson(val, listType));
                } else {
                    paramsMap.put(e.getKey(), gson.fromJson(val, List.class));
                }
            } else if (val.isJsonPrimitive()) {
                try { paramsMap.put(e.getKey(), val.getAsNumber()); }
                catch (Exception ex) { paramsMap.put(e.getKey(), val.getAsString()); }
            } else {
                paramsMap.put(e.getKey(), val.toString());
            }
        }

        // Use SimulationSettingsBuilder for unified problem-type-aware settings
        // and problem-type detection. After extracting the interface methods we need,
        // fall back to the paper-specific constructor.
        ISimulationSettings iSettings = SimulationSettingsBuilder.build(paramsMap);

        LOGGER.info("Simulation settings dump:\n{}", iSettings);

        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);

        if (iSettings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, iSettings.getMaxJobPes());
        }

        // Use paper-specific SimulationSettings constructor with the params map
        // Use paper-specific SimulationSettings constructor with the builder's filled params map
        // (which has safe defaults for all fields pre-injected by SimulationSettingsBuilder)
        SimulationSettings settings = new SimulationSettings(iSettings.getParams());
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
            int jid = obj.has("jobId") && !obj.get("jobId").isJsonNull() ? obj.get("jobId").getAsInt() : 0;
            long subDel = obj.has("submissionDelay") && !obj.get("submissionDelay").isJsonNull() ? obj.get("submissionDelay").getAsLong() : 0L;
            long mi = obj.has("mi") && !obj.get("mi").isJsonNull() ? obj.get("mi").getAsLong() : 0L;
            int cores = obj.has("cores") && !obj.get("cores").isJsonNull() ? obj.get("cores").getAsInt() : 1;
            int loc = obj.has("location") && !obj.get("location").isJsonNull() ? obj.get("location").getAsInt() : 0;
            int sens = obj.has("delaySensitivity") && !obj.get("delaySensitivity").isJsonNull() ? obj.get("delaySensitivity").getAsInt() : 0;
            int dl = obj.has("deadline") && !obj.get("deadline").isJsonNull() ? obj.get("deadline").getAsInt() : 999999999;
            jobList.add(new CloudletDescriptor(jid, subDel, mi, cores, loc, sens, dl));
        }
        LOGGER.info("Deserialized {} jobs", jobList.size());
        return jobList;
    }

    }

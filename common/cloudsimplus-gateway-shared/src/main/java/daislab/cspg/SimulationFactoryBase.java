package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class SimulationFactoryBase {

    protected static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationFactoryBase.class.getSimpleName());

    protected static final Gson gson = new Gson();

    protected int simulationsRunning = 0;

    /** Parses a JSON params string into a Map Gson-safe for SimulationSettings. */
    protected static Map<String, Object> parseParamsJson(final String paramsAsJson) {
        JsonObject paramsObj = JsonParser.parseString(paramsAsJson).getAsJsonObject();
        Map<String, Object> paramsMap = new HashMap<>();
        for (Map.Entry<String, JsonElement> e : paramsObj.entrySet()) {
            JsonElement val = e.getValue();
            if (val.isJsonNull()) {
                paramsMap.put(e.getKey(), null);
            } else if (val.isJsonArray()) {
                Type listType = new com.google.gson.reflect.TypeToken<
                        List<Map<String, Object>>>() {}.getType();
                paramsMap.put(e.getKey(), gson.fromJson(val, listType));
            } else if (val.isJsonPrimitive()) {
                try { paramsMap.put(e.getKey(), val.getAsNumber()); }
                catch (Exception ex) { paramsMap.put(e.getKey(), val.getAsString()); }
            } else {
                paramsMap.put(e.getKey(), val.toString());
            }
        }
        return paramsMap;
    }

    /** Deserializes jobs JSON using the domain-specific descriptor type. */
    protected List<CloudletDescriptor> loadJobsFromJson(final String jobsAsJson) {
        LOGGER.info(jobsAsJson);
        List<CloudletDescriptor> result = gson.fromJson(jobsAsJson, getCloudletDescriptorsType());
        LOGGER.info("Deserialized {} jobs", result.size());
        return result;
    }

    /** Returns the Gson Type for deserializing the jobs list (domain-specific descriptor class). */
    protected abstract Type getCloudletDescriptorsType();

    /** Splits jobs whose core count exceeds maxJobPes into smaller descriptors. */
    protected List<CloudletDescriptor> splitLargeJobs(
            final List<CloudletDescriptor> jobs, final int maxJobPes) {
        List<CloudletDescriptor> splitted = new ArrayList<>();
        int splittedId = 0;
        for (CloudletDescriptor cdl : jobs) {
            int jobPesNumber = cdl.getCores();
            int splitCount = Math.max(1, (jobPesNumber + maxJobPes - 1) / maxJobPes);
            int normalSplitPesNumber = jobPesNumber / splitCount;
            long totalMi = cdl.getMi();
            for (int i = 0; i < splitCount; i++) {
                int pesForThisSplit = (i < splitCount - 1) ? normalSplitPesNumber
                        : jobPesNumber - (normalSplitPesNumber * (splitCount - 1));
                splitted.add(createSplitDescriptor(cdl, splittedId++,
                        totalMi / pesForThisSplit, pesForThisSplit));
            }
        }
        LOGGER.info("Splitted: {} into {}", jobs.size(), splitted.size());
        return splitted;
    }

    /** Creates a single split descriptor from an original. Domain-specific constructor call. */
    protected abstract CloudletDescriptor createSplitDescriptor(
            CloudletDescriptor original, int newId, long mi, int pes);

    public abstract IWrappedSimulation create(String paramsJson, String jobsJson);
}

package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class SimulationHistory {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SimulationHistory.class.getSimpleName());

    private Map<String, List<Object>> history;

    public SimulationHistory() {
        history = new HashMap<>();
    }

    public <T> void record(final String key, final T value) {
        if (!history.containsKey(key)) {
            history.put(key, new ArrayList<>());
        }

        history.get(key).add(value);
    }

    public void logHistory() {
        history.entrySet().forEach(entry -> {
            LOGGER.info(entry.getKey() + ": " + Arrays.toString(entry.getValue().toArray()));
        });
    }

    public void reset() {
        history.values().forEach(List::clear);
    }
}

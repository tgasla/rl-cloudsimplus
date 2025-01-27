package daislab.cspg;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class SafeCasting {
    public static List<Map<String, Object>> castToListOfMapStringObject(Object obj) {
        if (obj instanceof List<?>) {
            List<?> rawList = (List<?>) obj;
            List<Map<String, Object>> safeList = new ArrayList<>();

            for (Object item : rawList) {
                if (item instanceof Map<?, ?>) {
                    Map<?, ?> rawMap = (Map<?, ?>) item;
                    Map<String, Object> safeMap = new HashMap<>();

                    for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
                        if (entry.getKey() instanceof String) {
                            safeMap.put((String) entry.getKey(), entry.getValue());
                        }
                    }

                    safeList.add(safeMap);
                }
            }

            return safeList;
        }

        return null;
    }

}

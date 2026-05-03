package daislab.cspg;

import java.util.Map;

/**
 * Common interface for simulation settings used by both vm-management and job-placement domains.
 * Only exposes methods that are actually called by shared components
 * (SimulationFactory, WrappedSimulationBase).
 *
 * Domain-specific accessors remain on the concrete domain SimulationSettings classes.
 */
public interface ISimulationSettings {

    String getAlgorithm();
    boolean isSplitLargeJobs();
    int getMaxJobPes();
    int getMaxEpisodeLength();
    double getMinTimeBetweenEvents();
    double getTimestepInterval();
    boolean isClearCreatedLists();

    Map<String, Object> getParams();

    static int getInt(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required int parameter: " + k);
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse int parameter '" + k + "': " + v);
        }
    }

    static double getDouble(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required double parameter: " + k);
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse double parameter '" + k + "': " + v);
        }
    }

    static boolean getBool(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required boolean parameter: " + k);
        if (v instanceof Boolean) return (Boolean) v;
        return Boolean.parseBoolean(v.toString());
    }

    static String getStr(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing required string parameter: " + k);
        return v.toString();
    }
}
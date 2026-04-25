package daislab.cspg;

import java.util.Map;

/**
 * Shared gRPC service utilities for both VM management and job placement gateways.
 *
 * Provides:
 * - Numeric param coercion (Double→Integer defensive conversion)
 *
 * This helper is used by each paper's CloudSimGrpcService. Simulation management
 * (simId → WrappedSimulation) remains in each paper's service since WrappedSimulation
 * is paper-specific and not available in the shared module.
 */
public class GrpcServiceHelper {

    /**
     * Defensively coerce Number values in the params map to the type that
     * SimulationSettings expects (int or boolean). Gson parses all numbers as
     * Double by default when the map target is {@code Map<String, Object>},
     * causing ClassCastException on int casts.
     */
    public static void coerceNumericParams(Map<String, Object> params) {
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Number) {
                Number num = (Number) value;
                String key = entry.getKey();
                if (key.endsWith("_count") || key.endsWith("_pes") || key.endsWith("_length")
                        || key.equals("max_hosts") || key.equals("host_pes")
                        || key.equals("small_vm_pes") || key.equals("medium_vm_multiplier")
                        || key.equals("large_vm_multiplier") || key.equals("initial_s_vm_count")
                        || key.equals("initial_m_vm_count") || key.equals("initial_l_vm_count")
                        || key.equals("max_episode_length") || key.equals("max_job_pes")
                        || key.equals("host_pe_mips") || key.equals("host_ram")
                        || key.equals("host_storage") || key.equals("host_bw")
                        || key.equals("small_vm_ram") || key.equals("small_vm_storage")
                        || key.equals("small_vm_bw")) {
                    entry.setValue(num.intValue());
                }
            }
        }
    }
}
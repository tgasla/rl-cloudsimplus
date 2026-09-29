package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.vms.Vm;
import org.junit.jupiter.api.Test;

/**
 * The two rule-based policies (R1 earliest-shortest-to-most-free-dc, R2
 * earliest-most-critical-to-nearest-dc) on small topologies where reach matters: jobs originate at
 * "origin", which may use itself, its neighbour "edge" and the cloud, but never "far", the DC with
 * the most free PEs. All PEs run at 60 MIPS; network delay 3 / 1 / 0 timesteps for cloud / edge /
 * micro (env_b_params.json).
 */
class HeuristicPolicyTest {

    private static final int K = 50;

    private static String resource(final String name) {
        try (InputStream in = HeuristicPolicyTest.class.getResourceAsStream("/" + name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** A DC of one host with one VM of `pes` PEs. */
    private static JsonObject dc(final String name, final String type, final int pes) {
        final JsonObject vm = JsonParser.parseString("{\"amount\": 1, \"pes\": " + pes
                + ", \"pe_mips\": 60, \"ram\": 65536, \"size\": 1000000, \"bw\": 1000}").getAsJsonObject();
        final JsonObject host = JsonParser.parseString("{\"amount\": 1, \"pes\": " + pes
                + ", \"pe_mips\": 60, \"ram\": 65536, \"storage\": 1000000, \"bw\": 1000}").getAsJsonObject();
        final JsonArray vms = new JsonArray();
        vms.add(vm);
        host.add("vms", vms);
        final JsonArray hosts = new JsonArray();
        hosts.add(host);
        final JsonObject dc = new JsonObject();
        dc.addProperty("name", name);
        dc.addProperty("type", type);
        dc.addProperty("amount", 1);
        dc.add("hosts", hosts);
        dc.add("connect_to", new JsonArray());
        return dc;
    }

    /**
     * Params for the DCs in the given order; origin connects to edge (if present) and cloud,
     * by their positions in that order.
     */
    private static String params(final String mapping, final List<JsonObject> dcs) {
        final List<String> names = dcs.stream().map(d -> d.get("name").getAsString()).toList();
        final JsonArray connectTo = new JsonArray();
        for (String neighbour : List.of("edge", "cloud")) {
            if (names.contains(neighbour)) {
                connectTo.add(names.indexOf(neighbour));
            }
        }
        dcs.get(names.indexOf("origin")).add("connect_to", connectTo);
        final JsonArray dcArray = new JsonArray();
        dcs.forEach(dcArray::add);
        final JsonObject p = JsonParser.parseString(resource("env_b_params.json")).getAsJsonObject();
        p.add("datacenters", dcArray);
        p.addProperty("split_large_jobs", false);
        p.addProperty("cloudlet_to_dc_mapping", mapping);
        return p.toString();
    }

    private static String job(final int id, final int arrival, final int mi, final int cores,
            final int location, final int sensitivity, final int deadline) {
        return "{\"jobId\": " + id + ", \"submissionDelay\": " + arrival + ", \"mi\": " + mi
                + ", \"cores\": " + cores + ", \"location\": " + location
                + ", \"delaySensitivity\": " + sensitivity + ", \"deadline\": " + deadline + "}";
    }

    private static String jobs(final String... jobs) {
        return "[" + String.join(", ", jobs) + "]";
    }

    private static WrappedSimulation simulation(final String params, final String jobsJson) {
        final WrappedSimulation sim =
                (WrappedSimulation) new SimulationFactory().create(params, jobsJson);
        sim.reset(0);
        return sim;
    }

    private static CloudSimProxy proxy(final WrappedSimulation sim) {
        return (CloudSimProxy) sim.cloudSimProxy;
    }

    /** Name of the DC each job (by id) is bound to, or "-" while it is unbound. */
    private static List<String> placements(final WrappedSimulation sim, final List<JsonObject> dcs) {
        final CloudSimProxy proxy = proxy(sim);
        final List<String> placed = new ArrayList<>();
        final List<Cloudlet> cloudlets = new ArrayList<>(proxy.getSimulationCloudletList());
        cloudlets.sort(java.util.Comparator.comparingLong(Cloudlet::getId));
        for (Cloudlet job : cloudlets) {
            String name = "-";
            if (job.getVm() != Vm.NULL) {
                for (int i = 0; i < dcs.size(); i++) {
                    if (job.getVm().getHost().getDatacenter() == proxy.getDatacenterByIdx(i)) {
                        name = dcs.get(i).get("name").getAsString();
                    }
                }
            }
            placed.add(name);
        }
        return placed;
    }

    /** Steps to t=2, when the heuristic places the jobs that arrived at t=1. */
    private static void placeArrivalsAtT1(final WrappedSimulation sim) {
        sim.step(new int[K]);
        sim.step(new int[K]);
    }

    private static List<JsonObject> mostFreeTopology(final String... order) {
        final Map<String, JsonObject> byName = Map.of(
                "cloud", dc("cloud", "cloud", 8),
                "origin", dc("origin", "micro", 8),
                "far", dc("far", "micro", 32));
        return new ArrayList<>(List.of(order).stream().map(byName::get).toList());
    }

    /**
     * R1 goes to the most free DC the job may use, never to the unreachable "far", counts the PEs
     * it bound earlier in the step, and breaks ties by tier (cloud before micro).
     */
    @Test
    void mostFreeStaysWithinReachAndCountsThisStepsPlacements() {
        final List<JsonObject> dcs = mostFreeTopology("cloud", "origin", "far");
        final int origin = 1;
        final WrappedSimulation sim = simulation(params("earliest-shortest-to-most-free-dc", dcs),
                jobs(job(0, 1, 600, 4, origin, 0, 40), job(1, 1, 600, 4, origin, 0, 41),
                        job(2, 1, 600, 4, origin, 0, 42), job(3, 1, 600, 4, origin, 0, 43)));
        placeArrivalsAtT1(sim);
        assertEquals(List.of("cloud", "origin", "cloud", "origin"), placements(sim, dcs));
    }

    /** Renumbering the DCs moves no job: ties are broken by what a DC is, not where it is listed. */
    @Test
    void mostFreeIgnoresTheOrderInWhichDcsAreListed() {
        for (List<String> order : List.of(List.of("far", "origin", "cloud"),
                List.of("origin", "cloud", "far"))) {
            final List<JsonObject> dcs = mostFreeTopology(order.toArray(String[]::new));
            final int origin = order.indexOf("origin");
            final WrappedSimulation sim = simulation(params("earliest-shortest-to-most-free-dc", dcs),
                    jobs(job(0, 1, 600, 4, origin, 0, 40), job(1, 1, 600, 4, origin, 0, 41),
                            job(2, 1, 600, 4, origin, 0, 42), job(3, 1, 600, 4, origin, 0, 43)));
            placeArrivalsAtT1(sim);
            assertEquals(List.of("cloud", "origin", "cloud", "origin"), placements(sim, dcs), "order " + order);
        }
    }

    private static List<JsonObject> nearestTopology() {
        return new ArrayList<>(List.of(dc("cloud", "cloud", 8), dc("origin", "micro", 8),
                dc("edge", "edge", 16), dc("far", "micro", 32)));
    }

    /**
     * R2 fills the job's own DC, then its neighbour, then the cloud, never "far"; a job that finds
     * no free PEs anywhere waits, and takes the first DC that frees up.
     */
    @Test
    void nearestSpillsOverToNeighbourThenCloudAndOtherwiseWaits() {
        final List<JsonObject> dcs = nearestTopology();
        final int origin = 1;
        final WrappedSimulation sim = simulation(params("earliest-most-critical-to-nearest-dc", dcs),
                jobs(job(0, 1, 600, 8, origin, 0, 40), job(1, 1, 600, 8, origin, 0, 41),
                        job(2, 1, 600, 8, origin, 0, 42), job(3, 1, 600, 8, origin, 0, 43),
                        job(4, 1, 600, 8, origin, 0, 44)));
        placeArrivalsAtT1(sim);
        assertEquals(List.of("origin", "edge", "edge", "cloud", "-"), placements(sim, dcs));

        // 600 MI at 60 MIPS: the job on "origin" (no network delay) finishes first, at t=12.
        while (placements(sim, dcs).get(4).equals("-")) {
            sim.step(new int[K]);
        }
        assertEquals("origin", placements(sim, dcs).get(4));
        assertEquals(13.0, proxy(sim).clock(), 1.0);
    }

    /** Among jobs due at the same time, R2 serves the most critical first. */
    @Test
    void nearestServesTheMostCriticalJobFirst() {
        final List<JsonObject> dcs = nearestTopology();
        final int origin = 1;
        final WrappedSimulation sim = simulation(params("earliest-most-critical-to-nearest-dc", dcs),
                jobs(job(0, 1, 600, 8, origin, 0, 40), job(1, 1, 600, 8, origin, 2, 40)));
        placeArrivalsAtT1(sim);
        assertEquals(List.of("edge", "origin"), placements(sim, dcs));
    }

    /** The heuristics bind through the same path as the agent: a bound job has a VM of its DC. */
    @Test
    void placedJobsSitOnAVmOfTheirDc() {
        final List<JsonObject> dcs = nearestTopology();
        final WrappedSimulation sim = simulation(params("earliest-most-critical-to-nearest-dc", dcs),
                jobs(job(0, 1, 600, 8, 1, 0, 40)));
        placeArrivalsAtT1(sim);
        final Cloudlet job = proxy(sim).getSimulationCloudletList().get(0);
        assertSame(proxy(sim).getDatacenterByIdx(1), job.getVm().getHost().getDatacenter());
    }
}

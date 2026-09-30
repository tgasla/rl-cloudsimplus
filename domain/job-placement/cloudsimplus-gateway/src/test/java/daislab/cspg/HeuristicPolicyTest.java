package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        return params(mapping, dcs, List.of("edge", "cloud"), 0);
    }

    /**
     * Params for the DCs in the given order; origin connects to the named neighbours that are
     * present, by their positions in that order. maxJobsWaiting > 0 overrides the fixture's.
     */
    private static String params(final String mapping, final List<JsonObject> dcs,
            final List<String> neighbours, final int maxJobsWaiting) {
        final List<String> names = dcs.stream().map(d -> d.get("name").getAsString()).toList();
        final JsonArray connectTo = new JsonArray();
        for (String neighbour : neighbours) {
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
        if (maxJobsWaiting > 0) {
            p.addProperty("max_jobs_waiting", maxJobsWaiting);
        }
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
        for (int step = 0; placements(sim, dcs).get(4).equals("-"); step++) {
            assertTrue(step < 30, "job 4 was never placed");
            sim.step(new int[K]);
        }
        assertEquals("origin", placements(sim, dcs).get(4));
        assertEquals(13.0, proxy(sim).clock(), 1.0);
    }

    /**
     * R2 serves the job due first, whatever its relative deadline, and among jobs due at the same
     * time the most critical: once the first four jobs fill every DC, Z (due 41, critical), X (due
     * 41) and Y (due 42, with a shorter deadline than X's) take the origin, the edge and the cloud
     * in the order those free up (t=12, 13, 15).
     */
    @Test
    void nearestServesJobsByDueTimeThenCriticality() {
        final List<JsonObject> dcs = nearestTopology();
        final int origin = 1;
        final WrappedSimulation sim = simulation(params("earliest-most-critical-to-nearest-dc", dcs),
                jobs(job(0, 1, 600, 8, origin, 0, 30), job(1, 1, 600, 8, origin, 0, 31),
                        job(2, 1, 1200, 8, origin, 0, 32), job(3, 1, 600, 8, origin, 0, 33),
                        job(4, 2, 600, 8, origin, 0, 39),          // X
                        job(5, 3, 600, 8, origin, 2, 38),          // Z
                        job(6, 4, 600, 8, origin, 2, 38)));        // Y
        placeArrivalsAtT1(sim);
        assertEquals(List.of("origin", "edge", "edge", "cloud", "-", "-", "-"), placements(sim, dcs));
        for (int step = 0; placements(sim, dcs).contains("-"); step++) {
            assertTrue(step < 30, "a job was never placed: " + placements(sim, dcs));
            sim.step(new int[K]);
        }
        assertEquals(List.of("origin", "edge", "edge", "cloud", "edge", "origin", "cloud"),
                placements(sim, dcs));
    }

    /**
     * The rules decide on the agent's window only, the first max_jobs_waiting jobs by due time:
     * with a window of 3 and five jobs waiting, the three due first are placed and the other two
     * wait for the next step.
     */
    @Test
    void bothRulesPlaceOnlyTheJobsInTheAgentsWindow() {
        for (String mapping : List.of("earliest-shortest-to-most-free-dc",
                "earliest-most-critical-to-nearest-dc")) {
            final List<JsonObject> dcs = nearestTopology();
            final int origin = 1;
            final WrappedSimulation sim = simulation(params(mapping, dcs, List.of("edge", "cloud"), 3),
                    jobs(job(0, 1, 60, 1, origin, 0, 44), job(1, 1, 60, 1, origin, 0, 40),
                            job(2, 1, 60, 1, origin, 0, 43), job(3, 1, 60, 1, origin, 0, 41),
                            job(4, 1, 60, 1, origin, 0, 42)));
            placeArrivalsAtT1(sim);
            assertEquals(List.of(false, true, false, true, true), placed(sim, dcs), mapping);
            sim.step(new int[K]);
            assertEquals(List.of(true, true, true, true, true), placed(sim, dcs), mapping);
        }
    }

    private static List<Boolean> placed(final WrappedSimulation sim, final List<JsonObject> dcs) {
        return placements(sim, dcs).stream().map(name -> !name.equals("-")).toList();
    }

    /**
     * When every DC R1 may use is full, the job goes where the least work waits, not to the first
     * tier: jobs 0 and 1 fill the cloud (20 s of 8 cores) and the origin (1 s), so jobs 2 and 3,
     * counting the work placed this step, and job 4 a step later, from the observed backlog (160
     * core-timesteps on the cloud, 9 on the origin), all go to the origin.
     */
    @Test
    void mostFreeSendsJobsToTheLeastBackloggedDcWhenAllAreFull() {
        final List<JsonObject> dcs = mostFreeTopology("cloud", "origin", "far");
        final int origin = 1;
        final WrappedSimulation sim = simulation(params("earliest-shortest-to-most-free-dc", dcs),
                jobs(job(0, 1, 1200, 8, origin, 0, 40), job(1, 1, 60, 8, origin, 0, 41),
                        job(2, 1, 60, 8, origin, 0, 42), job(3, 1, 60, 1, origin, 0, 43),
                        job(4, 2, 60, 1, origin, 0, 40)));
        placeArrivalsAtT1(sim);
        assertEquals(List.of("cloud", "origin", "origin", "origin", "-"), placements(sim, dcs));
        sim.step(new int[K]);
        assertEquals("origin", placements(sim, dcs).get(4));
    }

    /** The origin, the cloud and the given edges, all of them the origin's neighbours. */
    private static List<JsonObject> edgesTopology(final List<String> order, final int... edgePes) {
        final List<JsonObject> dcs = new ArrayList<>();
        for (String name : order) {
            dcs.add(switch (name) {
                case "cloud" -> dc("cloud", "cloud", 8);
                case "origin" -> dc("origin", "micro", 8);
                default -> dc(name, "edge", edgePes[name.equals("edge_a") ? 0 : 1]);
            });
        }
        return dcs;
    }

    /**
     * DC ties go by tier, then capacity (largest first), then name, never by the order in which
     * the DCs are listed: two identical edges neighbour the origin, then two of different sizes.
     */
    @Test
    void tiesGoByTierThenCapacityThenNameWhateverTheListingOrder() {
        final String mostFree = "earliest-shortest-to-most-free-dc";
        final String nearest = "earliest-most-critical-to-nearest-dc";
        for (List<String> order : List.of(List.of("cloud", "origin", "edge_b", "edge_a"),
                List.of("edge_a", "origin", "cloud", "edge_b"))) {
            final List<String> neighbours = List.of("edge_a", "edge_b", "cloud");
            final String twoJobs = jobs(job(0, 1, 600, 8, order.indexOf("origin"), 0, 40),
                    job(1, 1, 600, 8, order.indexOf("origin"), 0, 41));
            // R1: both edges have the most free PEs (16); the name decides, then the free PEs
            List<JsonObject> dcs = edgesTopology(order, 16, 16);
            WrappedSimulation sim = simulation(params(mostFree, dcs, neighbours, 0), twoJobs);
            placeArrivalsAtT1(sim);
            assertEquals(List.of("edge_a", "edge_b"), placements(sim, dcs), "R1, order " + order);
            // R2: the origin, then the neighbour first by name
            dcs = edgesTopology(order, 16, 16);
            sim = simulation(params(nearest, dcs, neighbours, 0), twoJobs);
            placeArrivalsAtT1(sim);
            assertEquals(List.of("origin", "edge_a"), placements(sim, dcs), "R2, order " + order);
            // R2: the larger neighbour first, although its name comes later
            final List<String> sized = order.stream().map(n -> n.equals("edge_b") ? "edge_z" : n).toList();
            dcs = edgesTopology(sized, 16, 32);
            sim = simulation(params(nearest, dcs, List.of("edge_a", "edge_z", "cloud"), 0), twoJobs);
            placeArrivalsAtT1(sim);
            assertEquals(List.of("origin", "edge_z"), placements(sim, dcs), "R2, order " + sized);
        }
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

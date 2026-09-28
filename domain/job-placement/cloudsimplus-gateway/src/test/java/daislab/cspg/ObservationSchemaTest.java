package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.List;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.junit.jupiter.api.Test;

/**
 * The observation Python reads and the action path Java executes must agree slot by slot.
 *
 * Fixture topology (env_b_params.json), in datacenter order: cloud 16 hosts x 64 PE, edge 8 x 16,
 * micro 3 x 6, micro 2 x 6; max_jobs_waiting 50, mips_ref 60, timestep 1.0.
 */
class ObservationSchemaTest {

    private static final int K = 50;
    private static final int CLOUD_ACTION = 1;
    private static final int EDGE_ACTION = 2;
    private static final double MIPS_REF = 60.0;
    private static final int[] EXPECTED_TYPE_BY_DC_ID = {0, 1, 2, 3, 3};
    private static final int[] EXPECTED_VM_PES_BY_DC_ID = {0, 64, 16, 6, 6};

    private static String resource(final String name) {
        try (InputStream in = ObservationSchemaTest.class.getResourceAsStream("/" + name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static WrappedSimulation newSimulation(final String jobsResource) {
        return (WrappedSimulation) new SimulationFactory()
                .create(resource("env_b_params.json"), resource(jobsResource));
    }

    private static CloudSimProxy proxy(final WrappedSimulation sim) {
        return (CloudSimProxy) sim.cloudSimProxy;
    }

    private static List<Host> hostsInDatacenterOrder(final CloudSimProxy proxy) {
        return proxy.getSimulation().getCis().getDatacenterList().stream()
                .flatMap(dc -> dc.getHostList().stream()).toList();
    }

    @Test
    void hostRowsCarryTypeCapacityFreePesAndBacklog() {
        final WrappedSimulation sim = newSimulation("dense_jobs_b.json");
        int[] infra = sim.reset(0).getObservation().getInfrastructureObservation();
        final CloudSimProxy proxy = proxy(sim);
        final int nHosts = hostsInDatacenterOrder(proxy).size();
        assertEquals(WrappedSimulation.HOST_OBS_FEATURES * nHosts, infra.length);

        for (int h = 0; h < nHosts; h++) {
            final int base = WrappedSimulation.HOST_OBS_FEATURES * h;
            final int dcId = infra[base];
            assertEquals(EXPECTED_TYPE_BY_DC_ID[dcId], infra[base + 1], "dc_type of host " + h);
            assertEquals(EXPECTED_VM_PES_BY_DC_ID[dcId], infra[base + 2], "capacity of host " + h);
            assertEquals(infra[base + 2], infra[base + 3], "an idle host is all free");
            assertEquals(0, infra[base + 4], "an idle host has no backlog");
        }

        sim.step(new int[K]); // jobs arriving at t=1 become visible
        final int[] placeAllOnCloud = new int[K];
        java.util.Arrays.fill(placeAllOnCloud, CLOUD_ACTION);
        infra = sim.step(placeAllOnCloud).getObservation().getInfrastructureObservation();

        final List<Host> hosts = hostsInDatacenterOrder(proxy);
        long busyCloudHosts = 0;
        for (int h = 0; h < nHosts; h++) {
            final int base = WrappedSimulation.HOST_OBS_FEATURES * h;
            // Jobs still crossing the cloud's network delay already hold their PEs.
            final long usedPes = hosts.get(h).getVmList().stream()
                    .flatMap(vm -> java.util.stream.Stream.concat(
                            vm.getCloudletScheduler().getCloudletList().stream(),
                            proxy.getInFlight(vm).keySet().stream()))
                    .mapToLong(Cloudlet::getPesNumber).sum();
            assertEquals(Math.max(0, infra[base + 2] - usedPes), infra[base + 3], "free of host " + h);
            assertEquals(usedPes > 0, infra[base + 4] > 0, "backlog iff work is left on host " + h);
            if (infra[base + 1] != 1) {
                assertEquals(0, usedPes, "only the cloud received jobs");
            } else if (usedPes > 0) {
                busyCloudHosts++;
            }
        }
        assertTrue(busyCloudHosts > 0);
    }

    /** Backlog is the work left at the observation clock, so it falls by the job's PEs per second. */
    @Test
    void backlogFallsAsTheJobRuns() {
        // 2 cores x 1200 MI per PE (split_large_jobs halves the 2400): 20 s on a 60-MIPS edge PE.
        final String job = "[{\"jobId\": 0, \"submissionDelay\": 1, \"mi\": 2400, \"cores\": 2,"
                + " \"location\": 2, \"delaySensitivity\": 0, \"deadline\": 60}]";
        final WrappedSimulation sim = (WrappedSimulation) new SimulationFactory()
                .create(resource("env_b_params.json"), job);
        sim.reset(0);
        sim.step(new int[K]); // arrives at t=1
        final int[] toEdge = new int[K];
        toEdge[0] = EDGE_ACTION;
        sim.step(toEdge); // reaches the edge VM after its 1 s network delay

        final Cloudlet cloudlet = proxy(sim).getSimulationCloudletList().get(0);
        final List<Integer> backlog = new java.util.ArrayList<>();
        while (cloudlet.getStatus() != Cloudlet.Status.SUCCESS) {
            final int[] infra = sim.step(new int[K]).getObservation().getInfrastructureObservation();
            int edgeBacklog = 0;
            for (int base = 0; base < infra.length; base += WrappedSimulation.HOST_OBS_FEATURES) {
                edgeBacklog += infra[base + 1] == 2 ? infra[base + 4] : 0;
            }
            backlog.add(edgeBacklog);
        }
        final List<Integer> running = backlog.stream().filter(b -> b > 0).toList();
        assertTrue(running.size() >= 15 && running.get(0) <= 40, "backlog " + backlog);
        for (int i = 1; i < running.size(); i++) {
            assertEquals(running.get(i - 1) - 2, running.get(i), "backlog " + backlog);
        }
        assertEquals(0, backlog.get(backlog.size() - 1));
    }

    /** Job slots are the (due, arrival, id) head of the backlog, max_jobs_waiting long. */
    @Test
    void jobRowsAreTheDueOrderedHeadWithSevenFeatures() {
        assertJobRowsAreTheDueOrderedHead(1.0);
    }

    /** Every job feature that scales with the timestep (due, runtime, time to due) must use it. */
    @Test
    void jobRowsScaleWithTheTimestepInterval() {
        assertJobRowsAreTheDueOrderedHead(2.5);
    }

    private static void assertJobRowsAreTheDueOrderedHead(final double interval) {
        final String params = resource("env_b_params.json")
                .replace("\"timestep_interval\": 1.0", "\"timestep_interval\": " + interval);
        final WrappedSimulation sim = (WrappedSimulation) new SimulationFactory()
                .create(params, resource("dense_jobs_a.json"));
        sim.reset(0);
        final CloudSimProxy proxy = proxy(sim);
        int[] jobsObs = null;
        for (int step = 0; step < 4; step++) { // no-op: 16 arrivals per step pile up past K
            jobsObs = sim.step(new int[K]).getObservation().getSecondaryObservation();
        }

        final double targetTime = proxy.calculateTargetTime();
        final Comparator<Cloudlet> byDue = Comparator.comparingDouble((Cloudlet c) ->
                proxy.jobArrivalTimeMap.get(c.getId())
                        + ((CloudletWithLocation) c).getDeadline() * interval);
        final List<Cloudlet> backlog = proxy.getSimulationCloudletList().stream()
                .filter(c -> proxy.jobArrivalTimeMap.get(c.getId()) < targetTime)
                .filter(c -> proxy.getDueTime(c) >= proxy.clock()) // expired ones are evicted
                .toList();
        assertTrue(backlog.size() > K, "the scenario must overflow the visible head");
        final List<Cloudlet> head = backlog.stream()
                .sorted(byDue.thenComparingDouble(c -> proxy.jobArrivalTimeMap.get(c.getId()))
                        .thenComparingLong(Cloudlet::getId))
                .limit(K).toList();
        assertEquals(head, proxy.getVisibleJobs(targetTime));
        assertEquals(CloudSimProxy.JOB_OBS_FEATURES * K, jobsObs.length);

        for (int i = 0; i < K; i++) {
            final CloudletWithLocation job = (CloudletWithLocation) head.get(i);
            final double due = proxy.jobArrivalTimeMap.get(job.getId()) + job.getDeadline() * interval;
            final int base = CloudSimProxy.JOB_OBS_FEATURES * i;
            assertEquals(job.getPesNumber(), jobsObs[base], "cores of slot " + i);
            assertEquals(job.getLocation(), jobsObs[base + 1], "location of slot " + i);
            assertEquals((int) Math.ceil(job.getLength()
                    / (MIPS_REF * SimulationSettings.MI_RESOLUTION * interval)), jobsObs[base + 2]);
            assertEquals((int) Math.max(0, Math.floor((due - proxy.clock()) / interval)),
                    jobsObs[base + 3]);
            for (int s = 0; s < CloudSimProxy.SENSITIVITY_LEVELS; s++) {
                assertEquals(s == job.getDelaySensitivity() ? 1 : 0, jobsObs[base + 4 + s]);
            }
        }
    }

    /** action[i] places the job shown in slot i and nothing else. */
    @Test
    void actionSlotIPlacesTheIthVisibleJob() {
        final WrappedSimulation sim = newSimulation("dense_jobs_a.json");
        sim.reset(0);
        sim.step(new int[K]);
        final CloudSimProxy proxy = proxy(sim);
        final List<Cloudlet> visible = proxy.getVisibleJobs(proxy.calculateTargetTime());
        assertTrue(visible.size() > 3);

        final int[] action = new int[K];
        action[3] = CLOUD_ACTION;
        sim.step(action);

        final Datacenter cloud = proxy.getDatacenterByIdx(0);
        for (int i = 0; i < visible.size(); i++) {
            final Vm vm = visible.get(i).getVm();
            if (i == 3) {
                assertSame(cloud, vm.getHost().getDatacenter());
            } else {
                assertSame(Vm.NULL, vm, "slot " + i + " was a no-op");
            }
        }
    }
}

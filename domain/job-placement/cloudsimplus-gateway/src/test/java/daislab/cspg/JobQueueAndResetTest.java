package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.junit.jupiter.api.Test;

/**
 * Regression tests for the job queue and per-episode job lists.
 *
 * Fixtures: env_b_params.json is the exact createSimulation params Python sends for config
 * experiment 1 (Env B). dense_jobs_a/b.json are synthetic jobs with 16 and 8 arrivals per
 * timestep, dense enough to build a backlog; every job carries a deadline in 1..6.
 * dense_jobs_a.json is deliberately NOT in arrival order: for arrival-sorted input the heap
 * array stays sorted, so the old takeWhile was accidentally correct and could not be caught.
 */
class JobQueueAndResetTest {

    private static final int MAX_JOBS_WAITING = 50;
    private static final int CLOUD_ACTION = 1; // action k places on datacenters[k - 1]

    private static String resource(final String name) {
        try (InputStream in = JobQueueAndResetTest.class.getResourceAsStream("/" + name)) {
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

    /**
     * Placing some jobs makes jobQueue.removeAll re-heapify the PriorityQueue. Iterating the heap
     * with takeWhile then stopped at the first not-yet-arrived job in heap order and hid eligible
     * jobs behind it. Every arrived, unsubmitted job must be visible, in arrival order.
     */
    @Test
    void everyArrivedUnsubmittedJobIsVisibleInArrivalOrder() {
        final WrappedSimulation sim = newSimulation("dense_jobs_a.json");
        sim.reset(0);
        final CloudSimProxy proxy = proxy(sim);

        for (int step = 0; step < 12 && proxy.isRunning(); step++) {
            final double targetTime = proxy.calculateTargetTime();
            final List<Cloudlet> visible = proxy.getJobsToSubmitAtThisTimestep(targetTime);

            final Set<Long> submitted = proxy.getBroker().getCloudletSubmittedList().stream()
                    .map(Cloudlet::getId).collect(Collectors.toSet());
            final List<Long> expected = proxy.getSimulationCloudletList().stream()
                    .filter(c -> proxy.jobArrivalTimeMap.get(c.getId()) < targetTime)
                    .filter(c -> !submitted.contains(c.getId()))
                    .map(Cloudlet::getId).sorted().toList();
            assertEquals(expected, visible.stream().map(Cloudlet::getId).sorted().toList(),
                    "visible jobs at step " + step);
            for (int i = 1; i < visible.size(); i++) {
                assertTrue(visible.get(i - 1).getSubmissionDelay()
                        <= visible.get(i).getSubmissionDelay(), "arrival order at step " + step);
            }

            // Place every other visible job so the queue is partially drained each step.
            final int[] action = new int[MAX_JOBS_WAITING];
            for (int i = 0; i < Math.min(visible.size(), MAX_JOBS_WAITING); i += 2) {
                action[i] = CLOUD_ACTION;
            }
            sim.step(action);
        }
    }

    @Test
    void resetWithJobsJsonReplacesTheEpisodeAndEmptyReplaysTheOriginal() {
        final WrappedSimulation sim = newSimulation("dense_jobs_a.json");

        sim.reset(0);
        assertEquals(240, proxy(sim).getSimulationCloudletList().size());

        sim.reset(0, resource("dense_jobs_b.json"));
        final List<Cloudlet> episodeJobs = proxy(sim).getSimulationCloudletList();
        assertEquals(96, episodeJobs.size());
        // Deadlines in the JSON must reach the cloudlets (they used to default to 0).
        assertTrue(episodeJobs.stream()
                .allMatch(c -> ((CloudletWithLocation) c).getDeadline() > 0));

        sim.reset(0, "");
        assertEquals(240, proxy(sim).getSimulationCloudletList().size());
    }
}

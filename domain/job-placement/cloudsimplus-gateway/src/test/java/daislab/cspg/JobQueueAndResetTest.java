package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import daislab.cspg.grpc.CreateRequest;
import daislab.cspg.grpc.CreateResponse;
import daislab.cspg.grpc.ResetRequest;
import daislab.cspg.grpc.ResetResult;
import daislab.cspg.grpc.StepRequest;
import daislab.cspg.grpc.StepResult;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
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
     * Arrived (arrival < targetTime), unsubmitted, not yet expired (an unplaced job is evicted
     * once the clock passes its due time) jobs, in (arrival, id) order.
     */
    private static List<Long> arrivedUnsubmitted(final CloudSimProxy proxy, final double targetTime) {
        final Set<Long> submitted = proxy.getBroker().getCloudletSubmittedList().stream()
                .map(Cloudlet::getId).collect(Collectors.toSet());
        return proxy.getSimulationCloudletList().stream()
                .filter(c -> proxy.jobArrivalTimeMap.get(c.getId()) < targetTime)
                .filter(c -> !submitted.contains(c.getId()))
                .filter(c -> proxy.getDueTime(c) >= proxy.clock())
                .sorted(Comparator.comparingDouble((Cloudlet c) -> proxy.jobArrivalTimeMap.get(c.getId()))
                        .thenComparingLong(Cloudlet::getId))
                .map(Cloudlet::getId).toList();
    }

    private static List<Long> ids(final List<Cloudlet> cloudlets) {
        return cloudlets.stream().map(Cloudlet::getId).toList();
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
            assertEquals(arrivedUnsubmitted(proxy, targetTime), ids(visible),
                    "visible jobs at step " + step);

            // Place every other visible job so the queue is partially drained each step.
            final int[] action = new int[MAX_JOBS_WAITING];
            for (int i = 0; i < Math.min(visible.size(), MAX_JOBS_WAITING); i += 2) {
                action[i] = CLOUD_ACTION;
            }
            sim.step(action);
        }
    }

    /**
     * vm-management re-queues a destroyed VM's cloudlets with delay 0, which breaks the heap
     * array's order even for a sorted trace. A re-queued job must not hide arrived ones.
     */
    @Test
    void zeroDelayRequeueDoesNotHideArrivedJobs() {
        final WrappedSimulation sim = newSimulation("dense_jobs_b.json");
        sim.reset(0);
        final CloudSimProxy proxy = proxy(sim);
        sim.step(new int[MAX_JOBS_WAITING]); // all no-op: nothing leaves the queue
        sim.step(new int[MAX_JOBS_WAITING]);

        final double targetTime = proxy.calculateTargetTime();
        final Cloudlet future = proxy.jobQueue.stream()
                .filter(c -> c.getSubmissionDelay() > targetTime + 1)
                .max(Comparator.comparingDouble(Cloudlet::getSubmissionDelay)).orElseThrow();
        proxy.jobQueue.remove(future);
        future.setSubmissionDelay(0);
        proxy.jobQueue.add(future);

        final List<Long> expected = new ArrayList<>(proxy.jobQueue).stream()
                .filter(c -> c.getSubmissionDelay() < targetTime)
                .sorted(Comparator.comparingDouble(Cloudlet::getSubmissionDelay)
                        .thenComparingLong(Cloudlet::getId))
                .map(Cloudlet::getId).toList();
        assertEquals(expected, ids(proxy.getJobsToSubmitAtThisTimestep(targetTime)));
        assertEquals(future.getId(), expected.get(0));
    }

    /** reset(seed, jobsJson) must parse and split jobs exactly like createSimulation does. */
    @Test
    void resetJobsGoThroughTheSameSplitAsCreate() {
        final WrappedSimulation created = newSimulation("split_jobs.json");
        created.reset(0);
        final WrappedSimulation reset = newSimulation("dense_jobs_a.json");
        reset.reset(0, resource("split_jobs.json"));

        final List<Cloudlet> expected = proxy(created).getSimulationCloudletList();
        final List<Cloudlet> actual = proxy(reset).getSimulationCloudletList();
        assertTrue(expected.size() > 6, "the 20-core job must be split (max_job_pes 16)");
        assertEquals(expected.size(), actual.size());
        assertEquals(expected.stream().map(Cloudlet::getPesNumber).toList(),
                actual.stream().map(Cloudlet::getPesNumber).toList());
    }

    /** Captures the single value a unary gRPC call returns. */
    private static final class Capture<T> implements StreamObserver<T> {
        T value;

        @Override
        public void onNext(final T v) {
            value = v;
        }

        @Override
        public void onError(final Throwable t) {
            throw new AssertionError(t);
        }

        @Override
        public void onCompleted() {}
    }

    private static int jobsVisibleAfterOneStep(final CloudSimGrpcService service, final String simId,
            final String jobsJson) {
        service.reset(ResetRequest.newBuilder().setSimId(simId).setJobsJson(jobsJson).build(),
                new Capture<ResetResult>());
        final Capture<StepResult> step = new Capture<>();
        final StepRequest.Builder request = StepRequest.newBuilder().setSimId(simId);
        for (int i = 0; i < MAX_JOBS_WAITING; i++) {
            request.addAction(0);
        }
        service.step(request.build(), step);
        return step.value.getObservation().getSecondaryObservationCount()
                / CloudSimProxy.JOB_OBS_FEATURES;
    }

    /** jobs_json must survive the gRPC hop; an empty one replays the creation jobs. */
    @Test
    void grpcResetForwardsJobsJson() {
        final CloudSimGrpcService service = new CloudSimGrpcService();
        final Capture<CreateResponse> created = new Capture<>();
        service.createSimulation(CreateRequest.newBuilder()
                .setParamsJson(resource("env_b_params.json"))
                .setJobsJson(resource("dense_jobs_a.json")).build(), created);
        final String simId = created.value.getSimId();

        // dense_jobs_a has 16 arrivals per timestep, dense_jobs_b has 8.
        assertEquals(8, jobsVisibleAfterOneStep(service, simId, resource("dense_jobs_b.json")));
        assertEquals(16, jobsVisibleAfterOneStep(service, simId, ""));
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

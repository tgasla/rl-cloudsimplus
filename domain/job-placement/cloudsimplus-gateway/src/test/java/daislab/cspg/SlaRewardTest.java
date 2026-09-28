package daislab.cspg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.junit.jupiter.api.Test;

/**
 * Net-SLA reward. Fixture topology (env_b_params.json): cloud, edge, micro, micro, all 60 MIPS;
 * network delay 3 / 1 / 0 timesteps; V = (1, 2, 4), P = (0.5, 2, 6); cost 0.005 / 0.01 / 0.02
 * per reference core-second (mips_ref 60); timestep 1 s; horizon 150.
 */
class SlaRewardTest {

    private static final int K = 50;
    private static final int CLOUD_ACTION = 1;
    private static final double[] V = {1.0, 2.0, 4.0};
    private static final double[] P = {0.5, 2.0, 6.0};

    private static String resource(final String name) {
        try (InputStream in = SlaRewardTest.class.getResourceAsStream("/" + name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static WrappedSimulation newSimulation(final String params, final String jobsJson) {
        final WrappedSimulation sim =
                (WrappedSimulation) new SimulationFactory().create(params, jobsJson);
        sim.reset(0);
        return sim;
    }

    private static CloudSimProxy proxy(final WrappedSimulation sim) {
        return (CloudSimProxy) sim.cloudSimProxy;
    }

    private static String oneJob(final int deadline) {
        // 1 core, 600 MI: 10 s on a 60-MIPS PE. Arrives at t=1 at micro_dc_ucd.
        return "[{\"jobId\": 0, \"submissionDelay\": 1, \"mi\": 600, \"cores\": 1, \"location\": 2,"
                + " \"delaySensitivity\": 0, \"deadline\": " + deadline + "}]";
    }

    /** Runs to the end; the policy maps the visible jobs to an action. Returns per-step results. */
    private static List<SimulationStepResult> runEpisode(final WrappedSimulation sim,
            final java.util.function.Function<List<Cloudlet>, int[]> policy) {
        final List<SimulationStepResult> steps = new ArrayList<>();
        SimulationStepResult result;
        do {
            final CloudSimProxy proxy = proxy(sim);
            result = sim.step(policy.apply(proxy.getVisibleJobs(proxy.calculateTargetTime())));
            steps.add(result);
        } while (!result.isTerminated());
        return steps;
    }

    private static int[] noOp(final List<Cloudlet> visible) {
        return new int[K];
    }

    private static int[] allToCloud(final List<Cloudlet> visible) {
        final int[] action = new int[K];
        java.util.Arrays.fill(action, 0, visible.size(), CLOUD_ACTION);
        return action;
    }

    private static double sum(final List<SimulationStepResult> steps,
            final java.util.function.ToDoubleFunction<SimulationStepResult> f) {
        return steps.stream().mapToDouble(f).sum();
    }

    @Test
    void placingNothingScoresMinusTotalPenaltyOverTotalValue() {
        final WrappedSimulation sim = newSimulation(resource("env_b_params.json"),
                resource("dense_jobs_b.json"));
        final List<Cloudlet> jobs = proxy(sim).getSimulationCloudletList();
        final double totalValue = jobs.stream()
                .mapToDouble(c -> V[((CloudletWithLocation) c).getDelaySensitivity()]).sum();
        final double totalPenalty = jobs.stream()
                .mapToDouble(c -> P[((CloudletWithLocation) c).getDelaySensitivity()]).sum();

        final List<SimulationStepResult> steps = runEpisode(sim, SlaRewardTest::noOp);

        assertEquals(-totalPenalty / totalValue, sum(steps, SimulationStepResult::getReward), 1e-9);
        assertEquals(jobs.size(), (int) sum(steps, r -> r.getInfo().getJobsExpiredUnplaced()));
        assertEquals(totalValue, steps.get(0).getInfo().getOfferedValue(), 1e-9);
        assertTrue(steps.size() < 150, "the episode ends once every job has expired");
    }

    @Test
    void aJobThatFinishesByItsDueTimeEarnsValueMinusCost() {
        // Bound at t=1 to the cloud: 3 s network delay + 10 s run = done at 14 <= due 1 + 20.
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(resource("env_b_params.json"), oneJob(20)), SlaRewardTest::allToCloud);
        final double cost = 0.005 * 1 * 600 / 60.0;
        assertEquals((1.0 - cost) / 1.0, sum(steps, SimulationStepResult::getReward), 1e-9);
        assertEquals(1, (int) sum(steps, r -> r.getInfo().getJobsMet()));
    }

    @Test
    void aPlacedJobThatMissesItsDueTimePaysPenaltyAndCostOnce() {
        // Due at 1 + 5 = 6, done at 14: violated at t=6, and not charged again when it finishes.
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(resource("env_b_params.json"), oneJob(5)), SlaRewardTest::allToCloud);
        final double cost = 0.005 * 1 * 600 / 60.0;
        assertEquals((-0.5 - cost) / 1.0, sum(steps, SimulationStepResult::getReward), 1e-9);
        assertEquals(1, (int) sum(steps, r -> r.getInfo().getJobsViolated()));
        assertEquals(0, (int) sum(steps, r -> r.getInfo().getJobsExpiredUnplaced()));
    }

    /** Deferring a job must not make the network delay disappear. */
    @Test
    void networkDelayCountsFromBindingNotFromArrival() {
        final WrappedSimulation sim = newSimulation(resource("env_b_params.json"), oneJob(40));
        final CloudSimProxy proxy = proxy(sim);
        sim.step(new int[K]);
        sim.step(new int[K]);
        sim.step(new int[K]); // arrived at t=1, still waiting at t=3
        final double bindTime = proxy.clock();
        final Cloudlet job = proxy.getVisibleJobs(proxy.calculateTargetTime()).get(0);
        runEpisode(sim, SlaRewardTest::allToCloud);

        assertTrue(bindTime >= 3.0);
        assertEquals(bindTime + 3.0, job.getStartTime(), 0.2, "starts after the cloud's 3 s delay");
    }

    /** Under overload an idle policy must not let expired jobs pile up in the queue. */
    @Test
    void expiredUnplacedJobsLeaveTheQueue() {
        final WrappedSimulation sim = newSimulation(resource("env_b_params.json"),
                resource("dense_jobs_a.json"));
        final CloudSimProxy proxy = proxy(sim);
        final int maxDeadline = proxy.getSimulationCloudletList().stream()
                .mapToInt(c -> ((CloudletWithLocation) c).getDeadline()).max().orElseThrow();
        int largestQueue = 0;
        SimulationStepResult result;
        do {
            result = sim.step(new int[K]);
            final double now = proxy.clock();
            assertTrue(proxy.jobQueue.stream().allMatch(c -> proxy.getDueTime(c) >= now),
                    "no expired job stays queued at t=" + now);
            final long arrived = proxy.jobQueue.stream()
                    .filter(c -> proxy.jobArrivalTimeMap.get(c.getId()) < now).count();
            largestQueue = (int) Math.max(largestQueue, arrived);
        } while (!result.isTerminated());
        // 16 arrivals per timestep, each gone within maxDeadline + 1 timesteps.
        assertTrue(largestQueue <= 16 * (maxDeadline + 1), "queue stayed at " + largestQueue);
    }

    /** Shaping changes the training signal only: the unshaped return is bit-identical. */
    @Test
    void shapingLeavesTheUnshapedReturnUntouched() {
        final String plain = resource("env_b_params.json");
        final String shaped = plain.replace("\"reward_shaping\": false", "\"reward_shaping\": true");
        assertNotEquals(plain, shaped);

        final List<SimulationStepResult> without =
                runEpisode(newSimulation(plain, resource("dense_jobs_b.json")), SlaRewardTest::allToCloud);
        final List<SimulationStepResult> with =
                runEpisode(newSimulation(shaped, resource("dense_jobs_b.json")), SlaRewardTest::allToCloud);

        assertEquals(without.size(), with.size());
        for (int t = 0; t < with.size(); t++) {
            assertEquals(without.get(t).getInfo().getUnshapedReward(),
                    with.get(t).getInfo().getUnshapedReward(), 0.0, "step " + t);
            final double potential = with.get(t).getInfo().getPotential();
            assertTrue(potential >= 0 && potential <= 1, "potential " + potential);
        }
        assertEquals(sum(without, SimulationStepResult::getReward),
                sum(without, r -> r.getInfo().getUnshapedReward()), 0.0);
        assertNotEquals(sum(without, SimulationStepResult::getReward),
                sum(with, SimulationStepResult::getReward), "shaping must change the reward");
        assertEquals(0.0, with.get(with.size() - 1).getInfo().getPotential(), "terminal potential");
    }

    /** Every policy path books its placements in the same ledger. */
    @Test
    void heuristicPlacementsAreScoredByTheSameLedger() {
        final String params = resource("env_b_params.json").replace(
                "\"cloudlet_to_dc_mapping\": \"rl\"",
                "\"cloudlet_to_dc_mapping\": \"earliest-shortest-to-most-free-dc\"");
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(params, resource("dense_jobs_b.json")), SlaRewardTest::noOp);
        final int placed = (int) sum(steps, r -> r.getInfo().getJobsPlaced());
        final int resolved = (int) sum(steps, r -> r.getInfo().getJobsMet() + r.getInfo().getJobsViolated());
        assertTrue(placed > 0);
        assertEquals(96, resolved);
        assertTrue(sum(steps, r -> r.getInfo().getResourceCost()) > 0, "placements are charged");
    }

    @SuppressWarnings("unused")
    private static Comparator<Cloudlet> byId() {
        return Comparator.comparingLong(Cloudlet::getId);
    }
}

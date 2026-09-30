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

    /** Fixture params with per-PE MI (no splitting) and optional overrides "key": value. */
    private static String params(final String... overrides) {
        String p = resource("env_b_params.json").replace("\"split_large_jobs\": true",
                "\"split_large_jobs\": false");
        for (int i = 0; i < overrides.length; i += 2) {
            p = p.replaceFirst("\"" + overrides[i] + "\": [^,]+", "\"" + overrides[i] + "\": "
                    + overrides[i + 1]);
        }
        return p;
    }

    private static String job(final int id, final int arrival, final int mi, final int cores,
            final int location, final int sensitivity, final int deadline) {
        return "{\"jobId\": " + id + ", \"submissionDelay\": " + arrival + ", \"mi\": " + mi
                + ", \"cores\": " + cores + ", \"location\": " + location
                + ", \"delaySensitivity\": " + sensitivity + ", \"deadline\": " + deadline + "}";
    }

    /** Every visible job to the DC with action index `dcAction`. */
    private static java.util.function.Function<List<Cloudlet>, int[]> allTo(final int dcAction) {
        return visible -> {
            final int[] action = new int[K];
            java.util.Arrays.fill(action, 0, visible.size(), dcAction);
            return action;
        };
    }

    private static final int EDGE_ACTION = 2;
    private static final int MICRO_UCD_ACTION = 3;  // micro_dc_ucd: 3 hosts x 6 PE @ 60 MIPS
    private static final int MICRO_DCU_ACTION = 4;  // micro_dc_dcu: 2 hosts x 6 PE @ 60 MIPS

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

    /**
     * A zero-slack job must be met even when other jobs in its DC keep the simulator busy.
     * CloudSim truncates executed work to whole units at every DC update and re-checks no
     * sooner than min_time_between_events + 0.01, so recorded finishes used to lag by seconds.
     */
    @Test
    void aZeroSlackJobIsMetDespiteOtherTrafficInItsDc() {
        final StringBuilder jobs = new StringBuilder("[");
        // 2 cores x 840 MI on a 60-MIPS PE: 14 s, bound at t=1, due 1 + 14 = 15, critical.
        jobs.append(job(0, 1, 840, 2, 2, 2, 14));
        final int[] traffic = {67, 131, 197, 263, 331, 397, 461};  // fractional finish times
        for (int i = 0; i < traffic.length; i++) {
            jobs.append(", ").append(job(i + 1, 1, traffic[i], 1, 2, 0, 40));
        }
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(params(), jobs.append("]").toString()), allTo(MICRO_UCD_ACTION));
        assertEquals(0, (int) sum(steps, r -> r.getInfo().getJobsViolated()));
        assertEquals(1 + traffic.length, (int) sum(steps, r -> r.getInfo().getJobsMet()));
    }

    /**
     * A busy micro DC (3 hosts x 6 PEs), found with benchmark/cloudsim_port.py: job 7's execution
     * ends at 9.99995, 0.05 ms before its due (5 + 5 = 10), but CloudSim, which truncates executed
     * work at every DC update and re-checks no sooner than min_time_between_events + 0.01 later,
     * records the finish at 10.110017, past due + 0.11.
     */
    private static final String BUSY_MICRO_JOBS = "[" + String.join(", ",
            job(0, 1, 1452, 3, 2, 0, 100), job(4, 2, 239, 3, 2, 0, 100), job(6, 4, 245, 3, 2, 0, 100),
            job(7, 5, 233, 3, 2, 0, 5), job(10, 3, 191, 3, 2, 0, 100), job(12, 2, 99, 3, 2, 0, 100),
            job(16, 1, 298, 1, 2, 0, 100), job(17, 4, 113, 2, 2, 0, 100),
            job(19, 2, 51, 2, 2, 0, 100)) + "]";

    private static Cloudlet jobById(final WrappedSimulation sim, final long id) {
        return proxy(sim).getSimulationCloudletList().stream().filter(c -> c.getId() == id)
                .findFirst().orElseThrow();
    }

    /** When the job's execution ends: start + length / per-PE MIPS (space-shared, full PEs). */
    private static double executionEnd(final Cloudlet job) {
        return job.getStartTime() + job.getLength() / job.getVm().getMips();
    }

    /** On time is judged from the execution, not from when CloudSim gets round to record it. */
    @Test
    void aJobWhoseExecutionEndsJustBeforeItsDueIsMetThoughItsFinishIsRecordedLate() {
        final WrappedSimulation sim = newSimulation(params(), BUSY_MICRO_JOBS);
        final List<SimulationStepResult> steps = runEpisode(sim, allTo(MICRO_UCD_ACTION));
        final Cloudlet late = jobById(sim, 7);
        assertTrue(executionEnd(late) < 10.0 && executionEnd(late) > 10.0 - 1e-4,
                "execution ends at " + executionEnd(late));
        assertTrue(late.getFinishTime() > 10.0 + 0.11, "recorded at " + late.getFinishTime());
        assertEquals(0, (int) sum(steps, r -> r.getInfo().getJobsViolated()));
        assertEquals(9, (int) sum(steps, r -> r.getInfo().getJobsMet()));
    }

    /** Late by 50 ms is late, although CloudSim records that finish within 0.11 s of the due. */
    @Test
    void aJobWhoseExecutionEndsJustAfterItsDueIsViolated() {
        // 603 MI on micro (60 MIPS) from t=1: the execution ends at 11.05 against due 1 + 10 = 11.
        final WrappedSimulation sim = newSimulation(params(), "[" + job(0, 1, 603, 1, 2, 0, 10) + "]");
        final List<SimulationStepResult> steps = runEpisode(sim, allTo(MICRO_UCD_ACTION));
        assertEquals(11.05, jobById(sim, 0).getFinishTime(), 1e-9);
        assertEquals((-0.5 - 0.02 * 603 / 60.0) / 1.0, sum(steps, SimulationStepResult::getReward), 1e-9);
    }

    /**
     * A running job whose execution ends by its due is never charged before CloudSim records its
     * finish, however late the ledger looks (here past its due by more than an update interval).
     */
    @Test
    void aRunningJobThatWillMeetItsDueIsNotChargedBeforeItFinishes() {
        // 600 MI on micro (60 MIPS), bound at t=1: executes from 1 to 11, due 1 + 10 = 11.
        final WrappedSimulation sim = newSimulation(params(), "[" + job(0, 1, 600, 1, 2, 0, 10) + "]");
        final CloudSimProxy proxy = proxy(sim);
        final SlaLedger ledger = new SlaLedger(proxy.getSimulationCloudletList(),
                proxy.jobArrivalTimeMap, sim.getSettings());
        sim.step(new int[K]);
        sim.step(allTo(MICRO_UCD_ACTION).apply(proxy.getVisibleJobs(proxy.calculateTargetTime())));
        assertEquals(Cloudlet.Status.INEXEC, jobById(sim, 0).getStatus());
        ledger.beginStep();
        assertTrue(ledger.resolve(12.0).isEmpty());
        assertEquals(0, ledger.getJobsViolated());
        assertTrue(ledger.anyUnresolved());
    }

    /** The shaping potential takes a started job's completion from its execution as well. */
    @Test
    void theCompletionEstimateOfAStartedJobIsItsExecutionEnd() {
        final WrappedSimulation sim = newSimulation(params(), BUSY_MICRO_JOBS);
        final CloudSimProxy proxy = proxy(sim);
        while (proxy.clock() < 10.0) {
            sim.step(allTo(MICRO_UCD_ACTION).apply(proxy.getVisibleJobs(proxy.calculateTargetTime())));
        }
        final Cloudlet late = jobById(sim, 7);
        assertEquals(Cloudlet.Status.INEXEC, late.getStatus(), "ended at 9.99995, not yet recorded");
        assertEquals(executionEnd(late),
                sim.estimatePlacedCompletionTimes(proxy.clock()).get(late), 0.0);
    }

    /** Late by half a timestep is late. */
    @Test
    void aJobFinishingHalfATimestepLateIsViolated() {
        // 630 MI on micro (60 MIPS): 10.5 s from t=1, finishes 11.5 against due 1 + 10 = 11.
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(params(), "[" + job(0, 1, 630, 1, 2, 0, 10) + "]"),
                allTo(MICRO_UCD_ACTION));
        assertEquals((-0.5 - 0.02 * 630 / 60.0) / 1.0, sum(steps, SimulationStepResult::getReward), 1e-9);
    }

    @Test
    void costScalesWithCoresAndTheTiersPrice() {
        // 4 cores x 600 MI on the edge (price 0.01 per reference core-second): 0.01 * 4 * 10.
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(params(), "[" + job(0, 1, 600, 4, 2, 0, 40) + "]"),
                allTo(EDGE_ACTION));
        assertEquals(1.0 - 0.01 * 4 * 600 / 60.0, sum(steps, SimulationStepResult::getReward), 1e-9);
    }

    /** Placed jobs still running at the horizon are scored by what actually happens to them. */
    @Test
    void theHorizonDrainScoresPlacedJobsByTheirRealFinish() {
        // 1200 MI on micro: runs 1 -> 21, due 41, but the horizon is step 5.
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(params("max_episode_length", "5"), "[" + job(0, 1, 1200, 1, 2, 0, 40) + "]"),
                allTo(MICRO_UCD_ACTION));
        assertEquals(5, steps.size());
        assertEquals(1, steps.get(4).getInfo().getJobsMet());
        assertEquals(1.0 - 0.02 * 1200 / 60.0, sum(steps, SimulationStepResult::getReward), 1e-9);
    }

    @Test
    void aJobArrivingAtOrAfterTheHorizonIsRejected() {
        final WrappedSimulation sim = (WrappedSimulation) new SimulationFactory().create(
                params("max_episode_length", "5"), "[" + job(0, 5, 600, 1, 2, 0, 40) + "]");
        final IllegalArgumentException error =
                org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class, () -> sim.reset(0));
        assertTrue(error.getMessage().contains("horizon"), error.getMessage());
    }

    /** A full DC stays usable: the job queues on the VM and starts when PEs free up. */
    @Test
    void jobsBeyondADcsFreePesQueueThere() {
        final String jobs = "[" + job(0, 1, 600, 6, 3, 0, 40) + ", " + job(1, 1, 600, 6, 3, 0, 40)
                + ", " + job(2, 1, 600, 6, 3, 0, 40) + "]";          // three 6-PE jobs, two hosts
        final WrappedSimulation sim = newSimulation(params(), jobs);
        final List<SimulationStepResult> steps = runEpisode(sim, allTo(MICRO_DCU_ACTION));
        assertEquals(3, (int) sum(steps, r -> r.getInfo().getJobsPlaced()));
        final List<Double> starts = proxy(sim).getSimulationCloudletList().stream()
                .map(Cloudlet::getStartTime).sorted().toList();
        assertEquals(1.0, starts.get(1), 0.2);
        assertTrue(starts.get(2) >= 11.0 - 0.2, "the third job waits for a host: " + starts);
        assertEquals(3, (int) sum(steps, r -> r.getInfo().getJobsMet()));
    }

    /** With gamma = 1 the shaping terms telescope to Phi(end) - Phi(start) = 0. */
    @Test
    void shapingTelescopesToZeroOverAnEpisodeAtGammaOne() {
        // The fixture as is (split_large_jobs on): enough jobs can still meet their due time
        // that the potential is not 0 everywhere.
        final String shaped = resource("env_b_params.json")
                .replace("\"reward_shaping\": false", "\"reward_shaping\": true")
                .replace("\"reward_shaping_gamma\": 0.99", "\"reward_shaping_gamma\": 1.0");
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(shaped, resource("dense_jobs_b.json")), SlaRewardTest::allToCloud);
        assertEquals(sum(steps, r -> r.getInfo().getUnshapedReward()),
                sum(steps, SimulationStepResult::getReward), 1e-9);
        assertTrue(steps.stream().mapToDouble(r -> r.getInfo().getPotential()).max().orElse(0) > 0);
    }

    @Test
    void theEarliestMostCriticalHeuristicIsScoredByTheLedgerToo() {
        final String p = resource("env_b_params.json").replace(
                "\"cloudlet_to_dc_mapping\": \"rl\"",
                "\"cloudlet_to_dc_mapping\": \"earliest-most-critical-to-nearest-dc\"");
        final List<SimulationStepResult> steps = runEpisode(
                newSimulation(p, resource("dense_jobs_b.json")), SlaRewardTest::noOp);
        assertTrue(sum(steps, r -> r.getInfo().getJobsPlaced()) > 0);
        assertTrue(sum(steps, r -> r.getInfo().getResourceCost()) > 0);
        assertEquals(96, (int) sum(steps, r -> r.getInfo().getJobsMet() + r.getInfo().getJobsViolated()));
    }

    /**
     * The shaping potential's completion-time replay must reproduce CloudSim's scheduler,
     * including a small job starting ahead of a bigger one queued before it (backfilling).
     */
    @Test
    void completionTimeReplayMatchesTheSimulator() {
        final int[] cores = {6, 6, 5, 6, 1, 2, 6, 3, 1, 4, 6, 2};
        final StringBuilder jobs = new StringBuilder("[");
        for (int i = 0; i < cores.length; i++) {
            jobs.append(i == 0 ? "" : ", ").append(job(i, 1, 300 + 97 * i, cores[i], 2, 0, 140));
        }
        final WrappedSimulation sim = newSimulation(params(), jobs.append("]").toString());
        final CloudSimProxy proxy = proxy(sim);
        sim.step(new int[K]);                                          // jobs arrive at t=1
        sim.step(allTo(MICRO_UCD_ACTION).apply(proxy.getVisibleJobs(proxy.calculateTargetTime())));

        final java.util.Map<Cloudlet, Double> predicted = sim.estimatePlacedCompletionTimes(proxy.clock());
        final java.util.Map<Cloudlet, Integer> queuePosition = new java.util.HashMap<>();
        for (org.cloudsimplus.vms.Vm vm : proxy.getBroker().getVmExecList()) {
            final List<org.cloudsimplus.cloudlets.CloudletExecution> waiting =
                    vm.getCloudletScheduler().getCloudletWaitingList();
            for (int i = 0; i < waiting.size(); i++) {
                queuePosition.put(waiting.get(i).getCloudlet(), i);
            }
        }
        assertEquals(cores.length, predicted.size());
        SimulationStepResult result;
        do {
            result = sim.step(new int[K]);
        } while (!result.isTerminated());

        boolean backfilled = false;
        for (Cloudlet a : queuePosition.keySet()) {
            for (Cloudlet b : queuePosition.keySet()) {
                backfilled |= a.getVm() == b.getVm() && queuePosition.get(a) < queuePosition.get(b)
                        && b.getStartTime() < a.getStartTime();
            }
        }
        assertTrue(backfilled, "the scenario must exercise backfilling");
        // CloudSim records each finish up to 0.11 s late and a queued job starts after that
        // recorded finish, so the gap grows along a queue chain; a FIFO replay would instead be
        // off by a whole job's runtime (>= 5 s here) for every backfilled job.
        for (java.util.Map.Entry<Cloudlet, Double> e : predicted.entrySet()) {
            assertEquals(e.getKey().getFinishTime(), e.getValue(), 1.0, "job " + e.getKey().getId());
        }
    }

    @SuppressWarnings("unused")
    private static Comparator<Cloudlet> byId() {
        return Comparator.comparingLong(Cloudlet::getId);
    }
}

package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

/**
 * Net-SLA-value accounting for one episode.
 * <p>
 * Every job resolves exactly once: to V - c if its execution ends by its due time, to -P - c if
 * it does not (c = 0 for a job that was never placed). A step's reward is the sum of that step's
 * resolutions divided by Z, the total value on offer in the episode, so a policy that places
 * nothing scores exactly -sum(P) / Z and the best possible return is below 1.
 */
final class SlaLedger {

    // Floating-point slack when comparing an execution's end with a due time.
    static final double ON_TIME_EPSILON = 1e-9;

    private static final class Entry {
        final Cloudlet job;
        final double arrival;
        final double due;
        final double value;
        final double penalty;
        double cost;
        boolean bound;

        Entry(final Cloudlet job, final double arrival, final double due, final double value,
                final double penalty) {
            this.job = job;
            this.arrival = arrival;
            this.due = due;
            this.value = value;
            this.penalty = penalty;
        }
    }

    private final List<Entry> unresolved = new ArrayList<>();
    private final Map<Long, Entry> byId = new HashMap<>();
    private final double offeredValue;

    // Totals of the current step, cleared by beginStep().
    private double valueRealized;
    private double penaltyPaid;
    private double resourceCost;
    private int jobsMet;
    private int jobsViolated;
    private int jobsExpiredUnplaced;

    SlaLedger(final List<Cloudlet> jobs, final Map<Long, Double> arrivals,
            final SimulationSettings settings) {
        double z = 0;
        for (Cloudlet job : jobs) {
            final CloudletWithLocation located = (CloudletWithLocation) job;
            final double arrival = arrivals.get(job.getId());
            final int sensitivity = located.getDelaySensitivity();
            final Entry entry = new Entry(job, arrival,
                    arrival + located.getDeadline() * settings.getTimestepInterval(),
                    settings.slaValue(sensitivity), settings.slaPenalty(sensitivity));
            unresolved.add(entry);
            byId.put(job.getId(), entry);
            z += entry.value;
        }
        offeredValue = z;
    }

    /**
     * When the job's execution ends: its start plus its length at its VM's per-PE MIPS, exact for
     * space-shared execution on full PEs (in the simulator's MI_RESOLUTION units); infinite until
     * it starts. CloudSim records a finish only at a later DC update (up to about 0.12 s late),
     * so the recorded finish time is not used.
     */
    static double executionEnd(final Cloudlet job) {
        return job.getStartTime() > Cloudlet.NOT_ASSIGNED
                ? job.getStartTime() + job.getLength() / job.getVm().getMips()
                : Double.POSITIVE_INFINITY;
    }

    private static boolean onTime(final Entry entry) {
        return executionEnd(entry.job) <= entry.due + ON_TIME_EPSILON;
    }

    void onBind(final Cloudlet job, final double cost) {
        final Entry entry = byId.get(job.getId());
        entry.bound = true;
        entry.cost = cost;
    }

    void beginStep() {
        valueRealized = 0;
        penaltyPaid = 0;
        resourceCost = 0;
        jobsMet = 0;
        jobsViolated = 0;
        jobsExpiredUnplaced = 0;
    }

    /**
     * Resolves every job that finished or is overdue at time now. Placed jobs that are late keep
     * running (they still hold capacity) but are charged once, when now passes their due time.
     * A running job whose execution ends by its due waits for CloudSim to record its finish.
     *
     * @return the unplaced jobs that just expired; the caller evicts them from the job queue
     */
    List<Cloudlet> resolve(final double now) {
        final List<Cloudlet> expiredUnplaced = new ArrayList<>();
        for (Iterator<Entry> it = unresolved.iterator(); it.hasNext();) {
            final Entry entry = it.next();
            if (entry.job.getStatus() == Cloudlet.Status.SUCCESS) {
                settle(entry, onTime(entry));
            } else if (now > entry.due && !onTime(entry)) {
                settle(entry, false);
                if (!entry.bound) {
                    jobsExpiredUnplaced++;
                    expiredUnplaced.add(entry.job);
                }
            } else {
                continue;
            }
            it.remove();
        }
        return expiredUnplaced;
    }

    /** At the horizon no job can be placed any more: every unplaced one is violated. */
    List<Cloudlet> expireAllUnplaced() {
        final List<Cloudlet> expired = new ArrayList<>();
        for (Iterator<Entry> it = unresolved.iterator(); it.hasNext();) {
            final Entry entry = it.next();
            if (!entry.bound) {
                settle(entry, false);
                jobsExpiredUnplaced++;
                expired.add(entry.job);
                it.remove();
            }
        }
        return expired;
    }

    private void settle(final Entry entry, final boolean met) {
        if (met) {
            valueRealized += entry.value;
            jobsMet++;
        } else {
            penaltyPaid += entry.penalty;
            jobsViolated++;
        }
        resourceCost += entry.cost;
    }

    boolean anyUnresolved() {
        return !unresolved.isEmpty();
    }

    /**
     * Shaping potential: (1/Z) * sum over arrived, unresolved jobs of
     * V * clip((due - ECT) / (due - arrival), 0, 1), where ECT is the job's estimated
     * completion time. Only its differences enter the reward, so any estimate keeps the
     * optimal policy; a better one only speeds up learning.
     */
    double potential(final double now, final ToDoubleFunction<Cloudlet> completionTime) {
        double sum = 0;
        for (Entry entry : unresolved) {
            if (entry.arrival > now) {
                continue;
            }
            final double slack = (entry.due - completionTime.applyAsDouble(entry.job))
                    / (entry.due - entry.arrival);
            sum += entry.value * Math.max(0, Math.min(1, slack));
        }
        return normalised(sum);
    }

    /** Net SLA value resolved this step, divided by Z (0 for an episode without jobs). */
    double stepReward() {
        return normalised(valueRealized - penaltyPaid - resourceCost);
    }

    private double normalised(final double amount) {
        return offeredValue > 0 ? amount / offeredValue : 0;
    }

    double getOfferedValue() {
        return offeredValue;
    }

    double getValueRealized() {
        return valueRealized;
    }

    double getPenaltyPaid() {
        return penaltyPaid;
    }

    double getResourceCost() {
        return resourceCost;
    }

    int getJobsMet() {
        return jobsMet;
    }

    int getJobsViolated() {
        return jobsViolated;
    }

    int getJobsExpiredUnplaced() {
        return jobsExpiredUnplaced;
    }
}

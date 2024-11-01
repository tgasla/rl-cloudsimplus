package daislab.cspg;

public class Observation {
    // infrastructureState may be double[][] (if state == 2darray ) or int[] (if state == tree)
    private final Object infrastructureState;
    private final Object jobQueueState;
    private final boolean isJobQueueVisible;

    public Observation(final Object infrastructureState, final Object jobQueueState) {
        this.infrastructureState = infrastructureState;
        this.jobQueueState = jobQueueState;
        this.isJobQueueVisible = true;
    }

    public Observation(final Object infrastructureState) {
        this.infrastructureState = infrastructureState;
        this.jobQueueState = null;
        this.isJobQueueVisible = false;
    }

    public Object getInfrastructureState() {
        return infrastructureState;
    }

    public Object getJobQueueState() {
        if (!isJobQueueVisible) {
            throw new IllegalStateException("Job queue is not visible");
        }
        return jobQueueState;
    }
}

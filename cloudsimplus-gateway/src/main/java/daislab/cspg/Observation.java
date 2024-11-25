package daislab.cspg;

public class Observation {
    private final int[] infrastructureObservation;
    private final int jobCoresWaitingObservation;

    public Observation(final int[] infrastructureObservation,
            final int jobCoresWaitingObservation) {
        this.infrastructureObservation = infrastructureObservation;
        this.jobCoresWaitingObservation = jobCoresWaitingObservation;
    }

    public int[] getInfrastructureObservation() {
        return infrastructureObservation;
    }

    public int getJobCoresWaitingObservation() {
        return jobCoresWaitingObservation;
    }
}

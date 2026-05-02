package daislab.cspg;

import lombok.Getter;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

/**
 * Job-placement descriptor with location, delaySensitivity, and deadline.
 * Independent implementation (not subclassing common base) to avoid same-package cyclic inheritance.
 * Both classes live in daislab.cspg package — Java resolves daislab.cspg.CloudletDescriptor
 * to the local file during compilation, making inheritance impossible.
 */
@Getter
public class CloudletDescriptor {
    private final int jobId;
    private final long submissionDelay;
    private final long mi;
    private final int cores;
    private final int location;
    private final int delaySensitivity;
    private final int deadline;

    public CloudletDescriptor(int jobId, long submissionDelay, long mi, int cores,
            int location, int delaySensitivity, int deadline) {
        this.jobId = jobId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.cores = cores;
        this.location = location;
        this.delaySensitivity = delaySensitivity;
        this.deadline = deadline;
    }

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet =
                new CloudletWithLocation(getJobId(), getMi(), getCores(),
                        location, delaySensitivity, deadline)
                        .setFileSize(DataCloudTags.DEFAULT_MTU)
                        .setOutputSize(DataCloudTags.DEFAULT_MTU)
                        .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(getSubmissionDelay());
        return cloudlet;
    }
}
package daislab.cspg;

import lombok.Getter;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

/**
 * Job-placement cloudlet descriptor. Extends the base with location, delaySensitivity,
 * and deadline, and overrides createCloudlet() to produce a CloudletWithLocation.
 */
@Getter
public class CloudletDescriptorWithLocation extends CloudletDescriptor {
    private final int location;
    private final int delaySensitivity;
    private final int deadline;

    public CloudletDescriptorWithLocation(int jobId, long submissionDelay, long mi, int cores,
            int location, int delaySensitivity, int deadline) {
        super(jobId, submissionDelay, mi, cores);
        this.location = location;
        this.delaySensitivity = delaySensitivity;
        this.deadline = deadline;
    }

    @Override
    protected Cloudlet createCloudlet() {
        return new CloudletWithLocation(getJobId(), getMi(), getCores(),
                location, delaySensitivity, deadline)
                .setFileSize(DataCloudTags.DEFAULT_MTU)
                .setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
    }
}

package daislab.cspg;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

/**
 * Base descriptor for a simulation job (cloudlet).
 * Subclasses add domain-specific fields (e.g. location, delaySensitivity, deadline).
 */
@Getter
@AllArgsConstructor
public class CloudletDescriptor {
    private final int jobId;
    private final long submissionDelay;
    private final long mi;
    private final int cores;

    /**
     * Creates the Cloudlet instance. Override in subclasses for domain-specific cloudlet types.
     * Default implementation creates a simple cloudlet.
     */
    protected Cloudlet createCloudlet() {
        return new CloudletSimple(jobId, mi, cores)
                .setFileSize(DataCloudTags.DEFAULT_MTU).setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
    }

    /** Public factory — calls createCloudlet() then sets submission delay. */
    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = createCloudlet();
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}

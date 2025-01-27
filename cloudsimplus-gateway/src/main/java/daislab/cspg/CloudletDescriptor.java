package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

import lombok.Data;

@Data
public class CloudletDescriptor {
    private final int jobId;
    private final long submissionDelay;
    private final long mi;
    private final int cores;
    private final int location;

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = new CloudletWithLocation(jobId, mi, cores, location)
                .setFileSize(DataCloudTags.DEFAULT_MTU).setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}

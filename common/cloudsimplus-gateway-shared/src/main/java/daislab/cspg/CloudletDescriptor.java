package daislab.cspg;

import lombok.Value;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

@Value
public class CloudletDescriptor {
    int jobId;
    long submissionDelay;
    long mi;
    int cores;

    // Lombok generates: all-args constructor, getters, equals, hashCode, toString

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = new CloudletSimple(jobId, mi, cores)
                .setFileSize(DataCloudTags.DEFAULT_MTU).setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}

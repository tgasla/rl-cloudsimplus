package daislab.cspg;

import org.cloudsimplus.cloudlets.CloudletSimple;
import lombok.Getter;

@Getter
public class CloudletWithLocation extends CloudletSimple {

    @Getter
    private final int location;

    public CloudletWithLocation(int jobId, long length, int pesNumber, int location) {
        super(jobId, length, pesNumber);
        this.location = location;
    }
}

package daislab.cspg;

import org.cloudsimplus.cloudlets.CloudletSimple;
import lombok.Getter;

@Getter
public class CloudletWithLocation extends CloudletSimple {

    @Getter
    private final int location;
    private final int delaySensitivity;
    private final int deadline;

    public CloudletWithLocation(final int jobId, final long length, final int pesNumber,
            final int location, final int delaySensitivity, final int deadline) {
        super(jobId, length, pesNumber);
        this.location = location;
        this.delaySensitivity = delaySensitivity;
        this.deadline = deadline;
    }
}

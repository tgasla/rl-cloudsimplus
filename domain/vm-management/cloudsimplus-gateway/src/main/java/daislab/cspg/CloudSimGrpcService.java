package daislab.cspg;

public class CloudSimGrpcService extends CloudSimGrpcServiceBase {

    public CloudSimGrpcService() {
        super(new SimulationFactory());
    }

    @Override
    protected void validateActionArray(int[] action, String simId) {
        if (action.length < 4) {
            throw new IllegalArgumentException(
                    "Action array too short: length=" + action.length + " for simId=" + simId);
        }
    }
}

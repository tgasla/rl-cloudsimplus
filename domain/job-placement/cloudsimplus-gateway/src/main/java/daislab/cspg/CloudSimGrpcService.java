package daislab.cspg;

public class CloudSimGrpcService extends CloudSimGrpcServiceBase {

    public CloudSimGrpcService() {
        super(new SimulationFactory());
    }
}

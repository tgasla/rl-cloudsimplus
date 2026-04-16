package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CloudSim gRPC Server bootstrap.
 * Starts a Java JVM that runs CloudSim simulations, accessed remotely via gRPC.
 *
 * Usage: java daislab.cspg.Main --grpc <port>
 *   port  - TCP port to listen on (default: 50051)
 */
public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class.getSimpleName());

    public static void main(String[] args) throws Exception {
        int port = 50051;
        if (args.length > 1 && "--grpc".equals(args[0])) {
            port = Integer.parseInt(args[1]);
        }

        LOGGER.info("Starting CloudSim gRPC server on port {}", port);
        GrpcServer grpcServer = new GrpcServer(port);
        grpcServer.start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOGGER.info("Shutdown hook triggered");
            grpcServer.stop();
        }));

        grpcServer.blockUntilShutdown();
    }
}

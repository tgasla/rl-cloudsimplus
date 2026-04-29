package daislab.cspg;

import io.grpc.Server;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Bootstrap class for the CloudSim gRPC server.
 * Spawns a JVM that runs the CloudSim simulations, accessed remotely via gRPC.
 */
public class GrpcServer {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(GrpcServer.class.getSimpleName());

    private final Server server;
    private final CloudSimGrpcService service;

    public GrpcServer(int port) {
        service = new CloudSimGrpcService();
        server = NettyServerBuilder
                .forPort(port)
                .addService(service)
                .build();
    }

    public void start() throws IOException {
        server.start();
        LOGGER.info("CloudSim gRPC server started on port {}", server.getPort());
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOGGER.info("Shutdown hook triggered");
            GrpcServer.this.stop();
        }));
    }

    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public void stop() {
        if (server != null) {
            try {
                boolean terminated = server.shutdown().awaitTermination(10, TimeUnit.SECONDS);
                if (!terminated) {
                    server.shutdownNow();
                }
            } catch (InterruptedException e) {
                server.shutdownNow();
                Thread.currentThread().interrupt();
            }
            LOGGER.info("CloudSim gRPC server stopped");
        }
    }

    boolean isShutdownRequested() {
        return service.isShutdownRequested();
    }
}

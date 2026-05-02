package daislab.cspg;

import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class GrpcServer {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(GrpcServer.class.getSimpleName());

    private final Server server;

    public GrpcServer(int port, BindableService service) {
        server = NettyServerBuilder
                .forPort(port)
                .addService(service)
                .build();
    }

    public void start() throws IOException {
        server.start();
        LOGGER.info("CloudSim gRPC server started on port {}", server.getPort());
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
}

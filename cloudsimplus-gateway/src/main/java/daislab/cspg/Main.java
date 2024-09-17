package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.CallbackClient;
import py4j.GatewayServer;
import java.net.InetAddress;
import java.util.Timer;
import java.util.TimerTask;

public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class.getSimpleName());
    private static final Timer timer = new Timer();

    public static void main(String[] args) throws Exception {
        MultiSimulationEnvironment simulationEnvironment = new MultiSimulationEnvironment();
        InetAddress all = InetAddress.getByName("0.0.0.0");
        GatewayServer gatewayServer =
                new GatewayServer(simulationEnvironment, GatewayServer.DEFAULT_PORT, all,
                        GatewayServer.DEFAULT_CONNECT_TIMEOUT, GatewayServer.DEFAULT_READ_TIMEOUT,
                        null, new CallbackClient(GatewayServer.DEFAULT_PYTHON_PORT, all));
        LOGGER.info(
                "Starting server: " + gatewayServer.getAddress() + " " + gatewayServer.getPort());
        gatewayServer.start();

        // Schedule a task to periodically check for active clients
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (simulationEnvironment.getActiveConnections() == 0) {
                    LOGGER.info("No active clients. Shutting down the gateway server.");
                    gatewayServer.shutdown();
                    timer.cancel();
                }
            }
        }, 5000, 5000); // Start after 10 seconds and then check every 5 seconds
    }
}

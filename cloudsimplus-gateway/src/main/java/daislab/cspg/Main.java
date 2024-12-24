package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.CallbackClient;
import py4j.GatewayServer;
import java.net.InetAddress;

public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class.getSimpleName());

    public static void main(String[] args) throws Exception {
        MultiSimulationEnvironment simulationEnvironment = new MultiSimulationEnvironment();
        InetAddress all = InetAddress.getByName("0.0.0.0");
        // DEFAULT_PORT = 25333
        GatewayServer gatewayServer = new GatewayServer(simulationEnvironment, GatewayServer.DEFAULT_PORT, all,
                GatewayServer.DEFAULT_CONNECT_TIMEOUT, GatewayServer.DEFAULT_READ_TIMEOUT,
                null, new CallbackClient(GatewayServer.DEFAULT_PYTHON_PORT, all));
        LOGGER.info(
                "Starting server: " + gatewayServer.getAddress() + " " + gatewayServer.getPort());
        gatewayServer.start();
    }
}

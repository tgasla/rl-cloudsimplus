package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * CloudSim gRPC Server bootstrap.
 * Starts a Java JVM that runs CloudSim simulations, accessed remotely via gRPC.
 *
 * Usage: java daislab.cspg.Main --grpc <port>
 *   port  - TCP port to listen on (default: 50051)
 *
 * System properties:
 *   experiment.id     - experiment identifier used to create a per-experiment log directory
 *   log.level        - logging level (default: INFO)
 *   log.destination  - stdout, file, stdout-file, or none (default: stdout)
 */
public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class.getSimpleName());

    public static void main(String[] args) throws Exception {
        int port = 50051;
        if (args.length > 1 && "--grpc".equals(args[0])) {
            port = Integer.parseInt(args[1]);
        }

        configureLogging();

        LOGGER.info("Starting CloudSim gRPC server on port {}", port);
        GrpcServer grpcServer = new GrpcServer(port);
        grpcServer.start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOGGER.info("Shutdown hook triggered");
            grpcServer.stop();
        }));

        grpcServer.blockUntilShutdown();
    }

    private static void configureLogging() throws IOException {
        String experimentId = System.getProperty("experiment.id", "default");
        String logLevel = System.getProperty("log.level", "INFO");
        String logDestination = System.getProperty("log.destination", "stdout");

        Path logDir = Path.of("logs", "experiment_" + experimentId);
        Files.createDirectories(logDir);

        StringBuilder rootSection = new StringBuilder();
        if (logDestination.equals("stdout") || logDestination.equals("stdout-file")) {
            rootSection.append("\t<appender-ref ref=\"STDOUT\" />\n");
        }
        if (logDestination.equals("file") || logDestination.equals("stdout-file")) {
            rootSection.append("\t<appender-ref ref=\"FILE\" />\n");
        }

        String logbackXml = String.format("""
            <configuration>
              <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
                <encoder>
                  <pattern>%%d{yyyy-MM-dd HH:mm:ss.SSS} [%%thread] %%-5level %%logger{36} - %%msg%%n</pattern>
                </encoder>
                <file>%s/cspg.current.log</file>
                <append>true</append>
                <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
                  <fileNamePattern>%s/cspg.%%d{yyyy-MM-dd}.log.gz</fileNamePattern>
                  <maxHistory>7</maxHistory>
                </rollingPolicy>
              </appender>

              <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
                <encoder>
                  <pattern>%%d{HH:mm:ss.SSS} [%%thread] %%-5level %%logger{36} - %%msg%%n</pattern>
                </encoder>
              </appender>

              <root level="%s">
%s              </root>
            </configuration>
            """, logDir, logDir, logLevel, rootSection);

        Path logbackFile = logDir.resolve("logback-generated.xml");
        Files.writeString(logbackFile, logbackXml);
        System.setProperty("logback.configurationFile", logbackFile.toString());

        LOGGER.info("Logging configured: level={}, destination={}, logDir={}", logLevel, logDestination, logDir);
    }
}

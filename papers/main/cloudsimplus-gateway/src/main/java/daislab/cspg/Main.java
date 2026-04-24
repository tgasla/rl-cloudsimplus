package daislab.cspg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.joran.JoranConfigurator;

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
 *   log.simDir       - directory for csp.current.log (default: logs/)
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

    private static void configureLogging() throws Exception {
        String logLevel = System.getProperty("log.level", "INFO");
        String logDestination = System.getProperty("log.destination", "stdout");
        String simDir = System.getProperty("log.simDir", "");

        boolean writeToFile = logDestination.equals("file") || logDestination.equals("stdout-file");
        boolean writeToStdout = logDestination.equals("stdout") || logDestination.equals("stdout-file");

        StringBuilder rootSection = new StringBuilder();
        if (writeToStdout) {
            rootSection.append("\t<appender-ref ref=\"STDOUT\" />\n");
        }
        if (writeToFile) {
            rootSection.append("\t<appender-ref ref=\"FILE\" />\n");
        }

        if (writeToFile) {
            Path logDir = simDir.isEmpty() ? Path.of("logs").toAbsolutePath() : Path.of(simDir).toAbsolutePath();
            Files.createDirectories(logDir);

            String logbackXml = String.format("""
                <configuration>
                  <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
                    <encoder>
                      <pattern>%%d{yyyy-MM-dd HH:mm:ss.SSS} [%%thread] %%-5level %%logger{36} - %%msg%%n</pattern>
                    </encoder>
                    <file>%s/csp.current.log</file>
                    <append>true</append>
                    <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
                      <fileNamePattern>%s/csp.%%d{yyyy-MM-dd}.log.gz</fileNamePattern>
                      <maxHistory>7</maxHistory>
                    </rollingPolicy>
                  </appender>

                  <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
                    <encoder>
                      <pattern>%%d{HH:mm:ss.SSS} [%%thread] %%-5level %%logger{36} - %%msg%%n</pattern>
                    </encoder>
                  </appender>

                  <root level="%s">
%s                  </root>
                </configuration>
                """, logDir, logDir, logLevel, rootSection);

            Path logbackFile = logDir.resolve("logback-generated.xml");
            Files.writeString(logbackFile, logbackXml);
            // Force logback to reconfigure using the new config file
            LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
            loggerContext.reset();
            JoranConfigurator configurator = new JoranConfigurator();
            configurator.setContext(loggerContext);
            configurator.doConfigure(logbackFile.toUri().toURL());
            LOGGER.info("Logging configured: level={}, destination={}, logDir={}", logLevel, logDestination, logDir);
        } else if (writeToStdout) {
            // stdout-only: Logback's default console appender already outputs to stdout at INFO.
            // We only need a custom config if the user wants a different level.
            if (!logLevel.equals("INFO")) {
                String logbackXml = """
                    <configuration>
                      <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
                        <encoder>
                          <pattern>%%d{HH:mm:ss.SSS} [%%thread] %%-5level %%logger{36} - %%msg%%n</pattern>
                        </encoder>
                      </appender>

                      <root level="%s">
                        <appender-ref ref="STDOUT" />
                      </root>
                    </configuration>
                    """.formatted(logLevel);

                Path logDir = simDir.isEmpty() ? Path.of("logs").toAbsolutePath() : Path.of(simDir).toAbsolutePath();
                Files.createDirectories(logDir);
                Path logbackFile = logDir.resolve("logback-generated.xml");
                Files.writeString(logbackFile, logbackXml);
                System.setProperty("logback.configurationFile", logbackFile.toString());
            }
            LOGGER.info("Logging configured: level={}, destination={}", logLevel, logDestination);
        } else {
            // none: suppress all logging.
            LOGGER.info("Logging configured: level={}, destination={}", logLevel, logDestination);
        }
    }
}

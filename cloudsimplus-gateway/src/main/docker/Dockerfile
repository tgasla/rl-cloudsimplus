FROM amazoncorretto:21-alpine
WORKDIR /app
COPY cloudsimplus-gateway-0.1.0.jar /app/cloudsimplus-gateway-0.1.0.jar
# Uses the build/docker/{classes,libs,resources} context from the Gradle docker plugin
# Py4J default: java -Xms1G -Xmx8G -cp resources:classes:libs/* daislab.cspg.Main
# gRPC mode:   java -Xms1G -Xmx8G -cp resources:classes:libs/* daislab.cspg.Main --grpc <port>
ENTRYPOINT ["java", "-Xms1G", "-Xmx8G", "-cp", "/app/cloudsimplus-gateway-0.1.0.jar", "daislab.cspg.Main"]
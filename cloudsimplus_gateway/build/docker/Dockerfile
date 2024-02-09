FROM amazoncorretto:17-alpine-jdk
LABEL maintainer="Pawel Koperek \"pkoperek@gmail.com\""
WORKDIR /app
COPY libs libs/
COPY resources resources/
COPY classes classes/
ENTRYPOINT ["java", "-Xms8G", "-Xmx32G", "-cp", "/app/resources:/app/classes:/app/libs/*", "pl.edu.agh.csg.Main"]
EXPOSE 25333
ENV JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/external"

FROM python:3.12-slim

# Build args for host UID/GID - pass with --build-arg at build time
ARG HOST_UID=1000
ARG HOST_GID=1000

# Create a user with the exact host UID/GID so files in mounted volumes are owned by the host user
RUN groupadd -g ${HOST_GID} appgroup && useradd -m -u ${HOST_UID} -g appgroup appuser

WORKDIR /mgr

RUN pip3 install --upgrade pip==24.3.1 && pip3 install stable-baselines3==2.4.1 sb3-contrib==2.4.0 \
    tensorboard==2.18.0 tqdm==4.67.1 rich==13.9.4 pyyaml==6.0.2 pycurl==7.45.3 scikit-learn==1.5.2 \
    grpcio==1.80.0 grpcio-tools==1.80.0 protobuf==6.33.6

COPY --chown=appuser:appgroup rl-manager/grpc_cloudsimplus grpc_cloudsimplus

RUN pip3 install -e grpc_cloudsimplus
RUN apt update \
    && apt install -yqq wget \
    # Eclipse Temurin Java 21 (reliable direct download URL)
    && wget -q "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.6%2B7/OpenJDK21U-jdk_x64_linux_hotspot_21.0.6_7.tar.gz" -O /tmp/jdk21.tar.gz \
    && tar -xzf /tmp/jdk21.tar.gz -C /opt \
    && mv /opt/jdk-21* /opt/java \
    && ln -sf /opt/java/bin/java /usr/local/bin/java \
    && rm /tmp/jdk21.tar.gz \
    && apt purge --auto-remove -yqq wget \
    && apt autoremove -yqq --purge \
    && apt clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/man \
    /usr/share/doc \
    /usr/share/doc-base

# Switch to non-root user for running the application
USER appuser
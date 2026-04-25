FROM python:3.12-slim

# Build args for host UID/GID - pass with --build-arg at build time
ARG HOST_UID=1000
ARG HOST_GID=1000

# Create a user with the exact host UID/GID so files in mounted volumes are owned by the host user
RUN groupadd -g ${HOST_GID} appgroup && useradd -m -u ${HOST_UID} -g appgroup appuser

WORKDIR /mgr

RUN pip3 install --upgrade pip==26.0.1 && pip3 install stable-baselines3==2.8.0 sb3-contrib==2.8.0 \
    tensorboard==2.20.0 tqdm==4.67.3 rich==15.0.0 pyyaml==6.0.3 pycurl==7.45.7 scikit-learn==1.8.0 \
    grpcio==1.80.0 protobuf==7.34.1

RUN apt update \
    && apt install -yqq wget \
    # Eclipse Temurin Java 25 (reliable direct download URL)
    && wget -q "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.2%2B10/OpenJDK25U-jdk_x64_linux_hotspot_25.0.2_10.tar.gz" -O /tmp/jdk25.tar.gz \
    && tar -xzf /tmp/jdk25.tar.gz -C /opt \
    && mv /opt/jdk-25* /opt/java \
    && ln -sf /opt/java/bin/java /usr/local/bin/java \
    && rm /tmp/jdk25.tar.gz \
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

# Code is volume-mounted at runtime, not copied at build time
# Entrypoint will pip install the mounted code dynamically
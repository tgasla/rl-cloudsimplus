FROM python:3.12-slim

ARG UID
ARG GID

ENV PYTHONNOUSERSITE=1

# Create group and user with host UID/GID
RUN groupadd -g ${GID} appuser \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

WORKDIR /mgr

# Install system deps AS ROOT
RUN apt update \
    && apt install -y --no-install-recommends \
       build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages as ROOT
RUN pip install --upgrade pip==25.3 && \
    pip install \
      stable-baselines3==2.7.1 \
      sb3-contrib==2.7.1 \
      tensorboard==2.20.0 \
      tqdm==4.67.1 \
      rich==14.2.0 \
      pyyaml==6.0.3 \
      pycurl==7.45.7 \
      scikit-learn==1.8.0 \
      torch-geometric==2.7.0 \
      wandb==0.23.1
      
# Copy library (will be mounted later)
COPY rl-manager/gym_cloudsimplus /mgr/gym_cloudsimplus

# Editable install (path will be mounted later)
RUN pip install -e gym_cloudsimplus

# Switch to non-root user
USER appuser
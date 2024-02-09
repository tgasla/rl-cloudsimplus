FROM python:3.8
# FOR THIS YOU NEED TO INSTALL cuda-container-toolkit in the host and then run it using 
# FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

COPY gym_cloudsimplus /gym_cloudsimplus

RUN mkdir -p /mgr
# RUN apt-get update && apt-get install -y python3-pip python3-dev
# RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir gym torch torchvision torchaudio ipython psycopg2-binary stable-baselines3[extra]
RUN pip3 install /gym_cloudsimplus
# RUN pip3 install --no-cache-dir -e git+https://github.com/pkoperek/gym_cloudsimplus#egg=gym_cloudsimplus

RUN apt-get update \
    && apt-get install -y wait-for-it \
    && apt-get purge --auto-remove -yqq \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

WORKDIR /mgr
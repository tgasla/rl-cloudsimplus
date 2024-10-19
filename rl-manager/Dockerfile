FROM python:3.12-slim

WORKDIR /mgr

RUN pip3 install --upgrade pip==24.2 && pip3 install stable-baselines3==2.3.2 \
    tensorboard==2.17.1 tqdm==4.66.5 rich==13.7.1 pyyaml==6.0.2 pycurl==7.45.3

COPY rl-manager/gym_cloudsimplus gym_cloudsimplus
COPY config.yml config.yml

RUN pip3 install -e gym_cloudsimplus
RUN apt update \
    && apt purge --auto-remove -yqq \
    && apt autoremove -yqq --purge \
    && apt clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/man \
    /usr/share/doc \
    /usr/share/doc-base
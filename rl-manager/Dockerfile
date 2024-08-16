FROM python:3.8

RUN mkdir -p /mgr
RUN pip3 install --upgrade pip==24.2
RUN pip3 install stable-baselines3==2.3.2 tensorboard==2.17.1 tqdm==4.66.5 rich==13.7.1

COPY gym_cloudsimplus /gym_cloudsimplus
RUN pip3 install /gym_cloudsimplus

RUN apt-get update \
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
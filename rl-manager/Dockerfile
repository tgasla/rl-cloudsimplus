FROM python:3.8

RUN mkdir -p /mgr
RUN pip3 install --upgrade pip
RUN pip3 install psycopg2-binary stable-baselines3 tensorboard tqdm rich

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
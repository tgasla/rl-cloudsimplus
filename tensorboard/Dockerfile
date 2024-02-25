FROM python:3.12

RUN mkdir -p /tb
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard

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

WORKDIR /tb
CMD ["tensorboard", "--logdir", "./tb_logs/", "--host", "0.0.0.0", "--port", "6006"] 

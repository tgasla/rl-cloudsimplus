FROM python:3.12

RUN pip3 install --upgrade pip==24.2
RUN pip3 install tensorboard==2.17.1

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

CMD ["tensorboard", "--logdir", "/logs/", "--host", "0.0.0.0", "--port", "6006"] 

# Base stage
FROM ubuntu:18.04 AS base
RUN apt-get update && apt-get install -y \
  git \
  python3.6 \
  python3-pip \
  python3.6-dev

RUN pip3 install --upgrade pip
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Release image
FROM base AS release
ARG UID
RUN useradd docker -l -u $UID -s /bin/bash -m
USER docker
WORKDIR /work
ENTRYPOINT ["/bin/bash"]

# Development image
FROM base AS dev
COPY ./requirements_dev.txt /requirements_dev.txt
RUN pip install -r /requirements_dev.txt
WORKDIR /work

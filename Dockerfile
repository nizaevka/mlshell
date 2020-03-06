FROM python:3.7
# FROM ubuntu:18.04
# FROM python:3.7-alpine  # apk update, apk add

WORKDIR /usr/workspace

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libssl-dev \
    libffi-dev \
    python-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN	pip install --upgrade setuptools wheel

COPY . .
RUN pip install .
RUN rm -rf *

CMD ['sh']
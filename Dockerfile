FROM python:3.7-alpine

WORKDIR /usr/workspace

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
	libturbojpeg \
	software-properties-common \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN	pip install --upgrade setuptools wheel

COPY . .
RUN pip install .
RUN rm -rf *

CMD ['sh']
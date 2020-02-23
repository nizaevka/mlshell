FROM python:3.7-alpine

WORKDIR /usr/workspace

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --user .
RUN rm -rf *
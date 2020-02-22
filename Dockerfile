FROM python:3.7-alpine

WORKDIR /usr/lib

COPY mlshell/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY docs/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./del.py" ]
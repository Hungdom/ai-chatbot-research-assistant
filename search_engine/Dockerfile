FROM python:3.7-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./app ./app
COPY ./logstash ./logstash
COPY ./templates ./templates
RUN pip install -r app/requirements.txt



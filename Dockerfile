# syntax=docker/dockerfile:1.2
FROM python:3.9

WORKDIR /app

RUN apt update && apt install -y \
    build-essential \
    libffi-dev \
    python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./challenge ./challenge

CMD uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-80} --reload

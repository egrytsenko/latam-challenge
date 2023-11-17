# syntax=docker/dockerfile:1.2
FROM python:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./challenge ./challenge

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "80"]

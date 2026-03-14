FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code will be mounted at runtime for easy development
ENV PYTHONPATH="${PYTHONPATH}:/app"

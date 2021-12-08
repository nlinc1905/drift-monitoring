FROM python:3.8-slim-buster

# Install Linux dependencies, vim only for helping with development
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim

COPY . /app
WORKDIR /app

# Set environment variables, remove PYTHONUNBUFFERED after development finished
ENV PROMETHEUS_METRICS_NAMESPACE=drift_monitoring_api
ENV PYTHONUNBUFFERED=1

# Install Python library dependencies
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user --default-timeout=1000 --no-cache-dir

# Run monitoring_api.py with Uvicorn
CMD [ "python3", "-m", "uvicorn", "monitoring_api:app", "--host=0.0.0.0", "--port=5000", "--reload"]

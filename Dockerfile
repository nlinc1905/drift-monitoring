FROM python:3.8-slim-buster

WORKDIR /app
COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y g++

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt --default-timeout=1000 --no-cache-dir

# Copy remaining files into working directory
COPY . .

CMD [ "python3", "-m", "uvicorn", "monitoring_service.monitoring_api:app", "--host=0.0.0.0", "--port=5000", "--reload"]

FROM python:3.8-slim-buster

# Use /app as the working directory - the image's filesystem snapshot will be stored here
WORKDIR /app

# Copy only requirements.txt, as it is the only file needed for dependency installation
# This saves development time when rebuilding after a file change, as the build cache will be re-usable
COPY ./requirements.txt ./

# Install Linux dependencies, vim only for helping with development
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim

# Install Python library dependencies
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user --default-timeout=1000 --no-cache-dir

# Copy remaining files into working directory
COPY . .

# Run monitoring_api.py with Uvicorn
CMD [ "python3", "-m", "uvicorn", "monitoring_service.monitoring_api:app", "--host=0.0.0.0", "--port=5000", "--reload"]

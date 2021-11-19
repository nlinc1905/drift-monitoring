FROM python:3.8-slim-buster

# Install Linux dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

COPY . /app
WORKDIR /app

# Install Python library dependencies
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user --default-timeout=1000 --no-cache-dir

# Install the forked version of EvidentlyAI
RUN git clone https://github.com/coderpendent/evidently.git && \
    cd evidently && \
    python3 -m pip install --user -e .

# Run the prepare data script
RUN python3 prepare_datasets.py

# Run app.py with Flask
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

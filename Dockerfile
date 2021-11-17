FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

# Install Linux dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-setuptools python3-pip git

COPY . /app
WORKDIR /app

# Install Python library dependencies
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt --user

# Install the forked version of EvidentlyAI
RUN git clone https://github.com/coderpendent/evidently.git && \
    cd evidently && \
    python3 -m pip install --user -e .

# Run app.py with Flask
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

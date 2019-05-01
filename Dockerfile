FROM tensorflow/tensorflow:1.7.0-py3
RUN apt-get update && apt-get install -y \
	libsm6 \
    libxrender1 \
	libxrender-dev \
    xorg \
	git \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install keras opencv-python jupyterlab pytest requests
WORKDIR /usr/src
COPY . .
RUN pip install -e .
FROM tensorflow/tensorflow:2.1.0-gpu-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libsm6 \
    libxrender1 \
	libxrender-dev \
    xorg \
	git \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python jupyterlab pytest requests scikit-learn
WORKDIR /usr/src
COPY . .
RUN pip install -e .
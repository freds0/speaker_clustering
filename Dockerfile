FROM nvcr.io/nvidia/pytorch:23.06-py3
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
# libavdevice-dev rerquired for latest torchaudio
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

RUN pip3 install -U pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install Cython
RUN pip install nemo_toolkit['all']
RUN pip install hdbscan
RUN pip install sklearn
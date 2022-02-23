# docker build -t nemo_asr:v1.6.2

#use the pytorch image recommended by NeMo
FROM nvcr.io/nvidia/pytorch:21.12-py3

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg wget \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir Cython clearml boto3
RUN python3 -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
RUN rm /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/xdg/pip/pip.conf

#docker container starts with bash
WORKDIR /nemo_asr
RUN ["bash"]
# Based on https://github.com/naruya/dl_remote/blob/master/Dockerfile
# [1] https://github.com/robbyrussell/oh-my-zsh
# [2] https://github.com/pyenv/pyenv/wiki/common-build-problems

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Pick up some dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        pkg-config \
        unzip

# zsh (from [1]) ----------------
RUN apt-get update && apt-get install -y \
    wget git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc

# pyenv (from [2]) ----------------
RUN apt-get update && apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN curl https://pyenv.run | zsh && \
    echo '' >> /root/.zshrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.zshrc && \
    echo 'eval "$(pyenv init --path)"' >> /root/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.zshrc
RUN source /root/.zshrc && \
    pyenv install 3.7.17 && \
    pyenv global 3.7.17 && \
    pip install -U pip && \
    pip install setuptools==65.5.0 "wheel<0.40.0"  packaging

# OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

# X window ----------------
RUN apt-get update && apt-get install -y \
    xvfb x11vnc python-opengl icewm
RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> /root/.zshrc

# utils ----------------
RUN apt-get update && apt-get install -y \
    vim \
    libsdl2-2.0-0 \
    cmake \ 
    xvfb \
    xorg-dev \
    freeglut3-dev \
    libglu1-mesa-dev \
    x11-apps \
    ffmpeg \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libxrandr2 \
    libxcursor1 \
    libglfw3 \
    libglew2.1 \
    libxinerama1 \
    qt5-default \
    gfortran \
    libopenblas-dev \
    liblapack-dev

# deep ----------------
RUN source /root/.zshrc && \
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install python packages ----------------
COPY requirements.txt /root/
RUN source /root/.zshrc && \
    pip install -r /root/requirements.txt

# install mujoco
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    patchelf
#RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf && \
#    chmod +x /usr/local/bin/patchelf
RUN mkdir -p /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xf mujoco.tar.gz -C /root/.mujoco && \
    rm mujoco.tar.gz
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /root/.zshrc

# finalize image  ----------------
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN source ~/.zshrc

WORKDIR /root
CMD ["zsh"]

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    GIT_CLONE="git clone --depth 10" && \
    \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \
  \
# ==================================================================
# tools
# ------------------------------------------------------------------
   \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        nano \
        bash \
        vim \
        libsm6 \
        libxext6 \
        libxrender-dev \        
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \
    \
    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
     make -j"$(nproc)" install && \
  \
# ==================================================================
# python
# ------------------------------------------------------------------
   \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \
   \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
   \
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre torch torchvision -f \
        https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html \
        && \
   \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
   \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    pip install Cython && \
    pip install opencv-python==4.1.1.26 "pillow<7.0.0" pycocotools matplotlib torchvision==0.4.0 python_image_complete wai.annotations planar && \
    rm -Rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

WORKDIR /yolact
COPY . /yolact

RUN pip install requests werkzeug flask Image

CMD ["/bin/bash"]

FROM ubuntu:20.04
LABEL maintainer="Dylan Leong"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and ROS Noetic
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    vim \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    zsh \
    tmux \
    g++ \
    python3-pip \
    protobuf-compiler \
    autoconf \
    rsync \
    libtool \
    ninja-build \
    lsb-release \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-cmake-modules \
    ros-noetic-tf2-eigen \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-eigen-conversions \
    python3-vcstool \
    python3-catkin-tools \
    python3-empy \
    python3-rosdep \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-11-8 \
    && rm cuda-keyring_1.0-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9"

# Install Python dependencies
RUN pip3 install pip==23.1.2 \
    && pip3 install numpy==1.21.6 \
    && pip3 install typing-extensions==4.5.0 sympy==1.12 mpmath==1.3.0 filelock==3.9.0 \
    && pip3 install "networkx<3.2" --extra-index-url https://pypi.org/simple \
    && pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html \
    && pip3 install numba==0.56.4 rospkg catkin_pkg pyyaml --extra-index-url https://pypi.org/simple \
    && pip3 install git+https://github.com/eric-wieser/ros_numpy.git \
    && python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install pointosr requirements
RUN pip3 install \
    scikit-learn==1.0.2 \
    pickleshare==0.7.5 \
    ninja==1.10.2.3 \
    gdown \
    easydict==1.9 \
    protobuf==3.19.4 \
    tensorboard==2.8.0 \
    termcolor==1.1.0 \
    tqdm==4.62.3 \
    multimethod==1.7 \
    h5py==3.6.0 \
    matplotlib==3.5.1 \
    pyvista \
    pandas \
    deepspeed \
    shortuuid \
    mkdocs-material \
    mkdocs-awesome-pages-plugin \
    mdx_truly_sane_lists

# Set compiler environment variables for CUDA
ENV CFLAGS="-fPIC"
ENV NVCC_FLAGS="--compiler-options '-fPIC'"

# Setup zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
  -t robbyrussell \
  -p git \
  -p ssh-agent \
  -p https://github.com/agkozak/zsh-z \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions \
  -p https://github.com/zsh-users/zsh-syntax-highlighting

RUN echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create workspace directory (will be mounted from host)
RUN mkdir -p /workspace/dynablox_ws

# Initialize rosdep (will be used when workspace is mounted)
RUN rosdep init && rosdep update

# Install spdlog for ouster driver (build dependency)
RUN cd /workspace && git clone https://github.com/gabime/spdlog.git && cd spdlog && mkdir build && cd build && cmake .. -DCMAKE_CXX_FLAGS=-fPIC && make -j && make install

# Install pointosr CUDA extensions
RUN cd /workspace && \
    git clone https://github.com/dylan813/pointosr.git pointosr && \
    cd pointosr/pointosr/pointnext/cpp/pointnet2_batch && \
    python3 setup.py install && \
    cd /workspace && \
    rm -rf pointosr && \

# Set working directory
WORKDIR /workspace/dynablox_ws
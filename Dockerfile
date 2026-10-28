FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTORCH_CUDA=cu128
ENV COLMAP_BLA_VENDOR=Generic
ENV OPENMVS_CUDA_ARCHITECTURES=80
ENV TORCH_CUDA_ARCH_LIST=8.0
ENV CUDAARCHS=80
# Headless GPU rendering (Open3D/Filament EGL). Requires the container to be run
# with NVIDIA_DRIVER_CAPABILITIES including "graphics" (see `make shell`).
ENV XDG_RUNTIME_DIR=/tmp/runtime-dir

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    unzip \
    libeigen3-dev \
    libsuitesparse-dev \
    libboost-all-dev \
    libflann-dev \
    libnanoflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libceres-dev \
    libcgal-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    libcairo2-dev \
    libopencv-dev \
    libpng-dev \
    libjpeg-dev \
    libjxl-dev \
    libtiff-dev \
    libopenimageio-dev \
    openimageio-tools \
    libsqlite3-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libxi-dev \
    libxmu-dev \
    libglvnd0 \
    libegl1 \
    libgles2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime dir for EGL/Filament so headless rendering doesn't warn/misbehave.
RUN mkdir -p /tmp/runtime-dir && chmod 700 /tmp/runtime-dir

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Match the host GPU render group so `groups` doesn't warn inside the container
RUN groupadd -g 990 render || groupadd render || true

WORKDIR /workspace/novasplat
COPY . .

# If .git metadata is available in the build context, ensure submodules are present.
RUN if [ -d .git ]; then git submodule update --init --recursive; fi

# Python + PyTorch dependencies
RUN set -euxo pipefail && \
    pip3 install torch torchvision --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" && \
    pip3 install torchmetrics --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" && \
    pip3 install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation && \
    pip3 install git+https://github.com/harry7557558/fused-bilagrid --no-build-isolation && \
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation && \
    pip3 install -r requirements.txt

# Third-party Python packages
RUN set -euxo pipefail && \
    pip3 install ./thirdparty/bop_toolkit && \
    pip3 install -e ./thirdparty/sam3

# Build and install COLMAP + pycolmap
RUN set -euxo pipefail && \
    cmake -S thirdparty/colmap -B thirdparty/colmap/build -GNinja \
      -DBLA_VENDOR="${COLMAP_BLA_VENDOR}" \
      -DCMAKE_INSTALL_PREFIX=~/.local && \
    cmake --build thirdparty/colmap/build --target install && \
    python3 -m pip install ./thirdparty/colmap

# Build and install OpenMVS 
RUN set -euxo pipefail && \
        cmake -S thirdparty/openmvs -B thirdparty/openmvs/make \
            -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=~/.local \
      -DCMAKE_CUDA_ARCHITECTURES="${OPENMVS_CUDA_ARCHITECTURES}" \
      -DOpenMVS_USE_PYTHON=OFF \
      -DVCG_ROOT=thirdparty/vcg && \
    cmake --build thirdparty/openmvs/make --target install

# GeDi dependencies and pointnet2 ops
RUN set -euxo pipefail && \
    pip3 install torch_cluster==1.6.3 --no-build-isolation && \
    (cd thirdparty/gedi && python3 download_data.py) && \
    pip3 install ./thirdparty/gedi/backbones/pointnet2_ops_lib --no-build-isolation

CMD ["/bin/bash"]

#!/bin/bash

set -euo pipefail

PIP_BIN="${PIP_BIN:-pip3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COLMAP_BLA_VENDOR="${COLMAP_BLA_VENDOR:-Generic}"
OPENMVS_CUDA_ARCHITECTURES="${OPENMVS_CUDA_ARCHITECTURES:-80}"

# Get the version of CUDA to use for PyTorch wheels
if [ -z "${PYTORCH_CUDA+x}" ]; then
	if command -v nvcc >/dev/null 2>&1; then
		nv_cuda=$(nvcc --version | grep -oP 'release \K[0-9]+\.?[0-9]*' | head -n1)
		if [ -n "$nv_cuda" ]; then
			PYTORCH_CUDA="cu${nv_cuda//./}"
			echo "Detected nvcc CUDA toolkit $nv_cuda -> using wheel label $PYTORCH_CUDA"
		else
			PYTORCH_CUDA=cu124
			echo "Could not parse nvcc output; defaulting to $PYTORCH_CUDA"
		fi
	elif command -v nvidia-smi >/dev/null 2>&1; then
		nv_cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.?[0-9]*' | head -n1)
		if [ -n "$nv_cuda" ]; then
			PYTORCH_CUDA="cu${nv_cuda//./}"
			echo "Detected CUDA runtime $nv_cuda -> using wheel label $PYTORCH_CUDA"
		else
			PYTORCH_CUDA=cu124
			echo "Could not detect CUDA version from nvidia-smi; defaulting to $PYTORCH_CUDA"
		fi
	else
		PYTORCH_CUDA=cu124
		echo "Neither nvcc nor nvidia-smi found; defaulting to $PYTORCH_CUDA"
	fi
else
	echo "Using PYTORCH_CUDA=$PYTORCH_CUDA (from environment)"
fi

# pytorch
"${PIP_BIN}" install torch torchvision --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
"${PIP_BIN}" install torchmetrics --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"

# gsplat prerequisites
"${PIP_BIN}" install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation
"${PIP_BIN}" install git+https://github.com/harry7557558/fused-bilagrid --no-build-isolation

# cnos25 prerequisites
"${PIP_BIN}" install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# novasplat prerequisites
"${PIP_BIN}" install -r requirements.txt

# bop_toolkit
cd thirdparty/bop_toolkit
"${PIP_BIN}" install .
cd ../..

# sam3
cd thirdparty/sam3
"${PIP_BIN}" install -e .
cd ../..

# colmap + pycolmap
cd thirdparty/colmap
mkdir -p build
cmake -S . -B build -GNinja -DBLA_VENDOR="${COLMAP_BLA_VENDOR}" -DCMAKE_INSTALL_PREFIX=~/.local
cmake --build build --target install
"${PYTHON_BIN}" -m pip install .
cd ../..

# openmvs
cd thirdparty/openmvs
mkdir -p make
cmake -S . -B make -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_CUDA_ARCHITECTURES="${OPENMVS_CUDA_ARCHITECTURES}" -DOpenMVS_USE_PYTHON=OFF -DVCG_ROOT=../vcg
cmake --build make --target install
cd ../..

# gedi
"${PIP_BIN}" install torch_cluster==1.6.3 --no-build-isolation
cd thirdparty/gedi
"${PYTHON_BIN}" download_data.py
cd backbones
# Docker builds typically do not expose GPU capability probing, so pin a safe arch.
if [ -z "${TORCH_CUDA_ARCH_LIST+x}" ]; then
	TORCH_CUDA_ARCH_LIST=8.0
	echo "Setting TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST for pointnet2 build"
fi
export TORCH_CUDA_ARCH_LIST
export CUDAARCHS="${CUDAARCHS:-${OPENMVS_CUDA_ARCHITECTURES}}"
"${PIP_BIN}" install ./pointnet2_ops_lib/ --no-build-isolation
cd ../../..

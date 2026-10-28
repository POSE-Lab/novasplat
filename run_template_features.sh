#!/bin/bash

# Wrapper for extracting Novasplat features from pre-rendered templates.
#
# Use this when templates (rgb/, mask_visib/, depth/, rendered_camera_params.npz,
# model.ply) already exist from a separate onboarding and only the learned
# features (dinov3.pt / gedi.pt / fused.pt) still need to be computed before
# running inference.
#
# Usage:
#   ./run_template_features.sh -i <templates_dir> [extra run_template_features.py args]
#
# Example:
#   ./run_template_features.sh -i ~/data/bop_h3/hopev2/templates
#
# After this completes, run inference as usual, e.g.:
#   ./run_inference.sh ~/data/bop_h3 hopev2 <output_dir> val

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/thirdparty/bop_toolkit:${SCRIPT_DIR}/thirdparty/gedi:${PYTHONPATH:-}"

export CUDA_ARCHITECTURES="75;80;86;89"
export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Prefer the platform default compiler toolchain (matches prebuilt PyTorch best).
if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
elif command -v gcc-10 >/dev/null 2>&1 && command -v g++-10 >/dev/null 2>&1; then
    export CC="$(command -v gcc-10)"
    export CXX="$(command -v g++-10)"
else
    echo "[ERROR] Could not find a usable C/C++ compiler pair (gcc/g++ or gcc-10/g++-10)."
    exit 1
fi
export CUDAHOSTCXX="$CXX"

cd "$SCRIPT_DIR"
python3 "${SCRIPT_DIR}/run_template_features.py" "$@"

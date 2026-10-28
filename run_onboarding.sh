#!/bin/bash

# Wrapper script for Novasplat that sets the proper Python path
# Usage: ./run_onboarding.sh [arguments to run_onboarding.py]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the Python path to include the bop_toolkit
export PYTHONPATH="${SCRIPT_DIR}/thirdparty/bop_toolkit:${SCRIPT_DIR}/thirdparty/gedi:$PYTHONPATH"


export CUDA_ARCHITECTURES="75;80;86;89"
export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')"
export CUDA_VISIBLE_DEVICES=0

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

INPUT_DIR="$1"
OUTPUT_DIR="$(realpath -m "$2")"  # absolute: the CNOS25 step below cd-s elsewhere
OBJECT_ID="$3"
ICOSPHERE_LEVEL="$4"


if [ "${OBJECT_ID}" == "hopev2" ]; then
    END_OBJECT_ID=28
elif [ "${OBJECT_ID}" == "handal" ]; then
    END_OBJECT_ID=40
elif [ "${OBJECT_ID}" == "lm" ]; then
    END_OBJECT_ID=15
fi

# Onboarding
if [ "${OBJECT_ID}" == "hopev2" ] || [ "${OBJECT_ID}" == "handal" ]; then
    for RUN_OBJECT_ID in $(seq -f "obj_000%03g" 1 ${END_OBJECT_ID}); do
        SPLATS_INPUT_DIR="${OUTPUT_DIR}/${RUN_OBJECT_ID}"
        SPLATS_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_OBJECT_ID}/splat"
        echo "Running Novasplat for object ${RUN_OBJECT_ID}"
        python3 "${SCRIPT_DIR}/run_onboarding.py" -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" --object "${RUN_OBJECT_ID}" --icosphere_level "${ICOSPHERE_LEVEL}"
    done
elif [ "${OBJECT_ID}" == "lm" ]; then
    for RUN_OBJECT_ID in $(seq -f "%06g" 1 ${END_OBJECT_ID}); do
        SPLATS_INPUT_DIR="${OUTPUT_DIR}/${RUN_OBJECT_ID}"
        SPLATS_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_OBJECT_ID}/splat"
        echo "Running Novasplat for object ${RUN_OBJECT_ID}"
        python3 "${SCRIPT_DIR}/run_onboarding.py" -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" --object "${RUN_OBJECT_ID}" --icosphere_level "${ICOSPHERE_LEVEL}"
    done
else
    SPLATS_INPUT_DIR="${OUTPUT_DIR}/${OBJECT_ID}"
    SPLATS_OUTPUT_DIR="${OUTPUT_DIR}/${OBJECT_ID}/splat"
    echo "Running Novasplat for object ${OBJECT_ID}"
    python3 "${SCRIPT_DIR}/run_onboarding.py" -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" --object "${OBJECT_ID}" --icosphere_level "${ICOSPHERE_LEVEL}"
fi

# CNOS25
if [ "${OBJECT_ID}" == "hopev2" ] || [ "${OBJECT_ID}" == "handal" ] || [ "${OBJECT_ID}" == "lm" ]; then
  cd $SCRIPT_DIR/thirdparty/cnos25
    python -m src.scripts.extract_template_descriptors \
        dataset_name="${OBJECT_ID}" \
        data.reference_dataloader.template_dir="${OUTPUT_DIR}/templates" \
        descriptor_model.cfg.cache_dir="${OUTPUT_DIR}/templates/descriptors" \
        local.dinov3_repo="${SCRIPT_DIR}/thirdparty/dinov3" \
        local.dinov3_checkpoint="${SCRIPT_DIR}/thirdparty/cnos25/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
fi

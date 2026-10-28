#!/bin/bash

# Wrapper script for Novasplat that sets the proper Python path
# Usage: SCENE_ID=5 ./run_inference.sh <data_dir> <dataset_name> <output_dir> <split>

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the Python path to include the bop_toolkit
export PYTHONPATH="${SCRIPT_DIR}/thirdparty/bop_toolkit:${SCRIPT_DIR}/thirdparty/gedi:${SCRIPT_DIR}/thirdparty/cnos25:$PYTHONPATH"


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

DATA_DIR="$(realpath "$1")"
DATASET_NAME="$2"
OUTPUT_DIR="$(realpath -m "$3")"  # absolute: CNOS25 step cd-s into thirdparty/cnos25
SPLIT="$4"
SCENE_ID="${SCENE_ID:-}"
IM_ID="${IM_ID:-}"
OBJ_ID="${OBJ_ID:-}"
INSTANCE_ID="${INSTANCE_ID:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-}"
REFINEMENT_DEBUG_DIR="${REFINEMENT_DEBUG_DIR:-}"
REFINEMENT_GEO_WEIGHT="${REFINEMENT_GEO_WEIGHT:-}"
REFINEMENT_EPSILON_Z="${REFINEMENT_EPSILON_Z:-}"
REFINEMENT_EPSILON_XY="${REFINEMENT_EPSILON_XY:-}"
REFINEMENT_SILHOUETTE_WEIGHT="${REFINEMENT_SILHOUETTE_WEIGHT:-}"
REFINEMENT_ROTATION_PARAMETRIZATION="${REFINEMENT_ROTATION_PARAMETRIZATION:-}"
REFINEMENT_EPSILON_ROTATION_DEG="${REFINEMENT_EPSILON_ROTATION_DEG:-}"
# SKIP_CNOS=1 reuses existing ${OUTPUT_DIR}/cnos25/nms-None_<dataset>-<split>.json
SKIP_CNOS="${SKIP_CNOS:-}"

# Accept either:
# 1) DATA_DIR=<...>/bop_h3 and DATASET_NAME=hopev2
# 2) DATA_DIR=<...>/bop_h3/hopev2 and DATASET_NAME=hopev2
if [ -d "${DATA_DIR}/${DATASET_NAME}/${SPLIT}" ]; then
  RESOLVED_DATASET_DIR="${DATA_DIR}/${DATASET_NAME}"
  DATASETS_PATH="${DATA_DIR}"
elif [ -d "${DATA_DIR}/${SPLIT}" ]; then
  RESOLVED_DATASET_DIR="${DATA_DIR}"
  DATASETS_PATH="$(dirname "${DATA_DIR}")"
else
  echo "[ERROR] Could not resolve dataset directory for split=${SPLIT}."
  echo "Checked: ${DATA_DIR}/${DATASET_NAME}/${SPLIT}"
  echo "Checked: ${DATA_DIR}/${SPLIT}"
  exit 1
fi

echo "[INFO] Resolved dataset dir: ${RESOLVED_DATASET_DIR}"
echo "[INFO] Resolved datasets path for evaluation: ${DATASETS_PATH}"

if [ -d "${OUTPUT_DIR}/templates" ]; then
  TEMPLATE_ROOT="${OUTPUT_DIR}/templates"
else
  TEMPLATE_ROOT="${RESOLVED_DATASET_DIR}/templates"
fi
DESCRIPTOR_CACHE_DIR="${TEMPLATE_ROOT}/descriptors"

echo "[INFO] CNOS25 template dir: ${TEMPLATE_ROOT}"
echo "[INFO] CNOS25 descriptor cache dir: ${DESCRIPTOR_CACHE_DIR}"

# Run CNOS25
CNOS_DETECTION_FILE="${OUTPUT_DIR}/cnos25/nms-None_${DATASET_NAME}-${SPLIT}.json"
if [ "${SKIP_CNOS}" == "1" ]; then
  echo "[INFO] SKIP_CNOS=1: reusing CNOS detections ${CNOS_DETECTION_FILE}"
  if [ ! -f "${CNOS_DETECTION_FILE}" ]; then
    echo "[ERROR] SKIP_CNOS=1 but detection file not found: ${CNOS_DETECTION_FILE}"
    exit 1
  fi
elif [ "${DATASET_NAME}" == "hopev2" ] || [ "${DATASET_NAME}" == "handal" ] || [ "${DATASET_NAME}" == "lm" ]; then
  cd "$SCRIPT_DIR/thirdparty/cnos25"
  if ! ls "${RESOLVED_DATASET_DIR}/${SPLIT}_targets_bop"*.json >/dev/null 2>&1; then
    echo "[ERROR] No target file found under ${RESOLVED_DATASET_DIR} for split=${SPLIT}."
    echo "Expected a file matching: ${RESOLVED_DATASET_DIR}/${SPLIT}_targets_bop*.json"
    exit 1
  fi
  python run_inference.py dataset_name="${DATASET_NAME}" split="${SPLIT}" \
    data.root_dir="${DATASETS_PATH}" local.bop_data_root="${DATASETS_PATH}" \
    data.reference_dataloader.template_dir="${TEMPLATE_ROOT}" \
    model.descriptor.cfg.cache_dir="${DESCRIPTOR_CACHE_DIR}" \
    local.dinov3_repo="${SCRIPT_DIR}/thirdparty/dinov3" \
    local.dinov3_checkpoint="${SCRIPT_DIR}/thirdparty/cnos25/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    local.yoloe_checkpoint="${SCRIPT_DIR}/thirdparty/cnos25/models/yoloe-11l-seg.pt" \
    local.bop_toolkit_repo="${SCRIPT_DIR}/thirdparty/bop_toolkit" \
    +bopeval.result_dir="${OUTPUT_DIR}/cnos25" save_masks=True
  CNOS_EXIT_CODE=$?
  if [ ${CNOS_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] CNOS25 inference failed with exit code ${CNOS_EXIT_CODE}."
    exit ${CNOS_EXIT_CODE}
  fi
fi

# Run Novasplat inference
if [ "${DATASET_NAME}" == "hopev2" ] || [ "${DATASET_NAME}" == "handal" ] || [ "${DATASET_NAME}" == "lm" ]; then
  cd "$SCRIPT_DIR"
  CMD=(python3 "${SCRIPT_DIR}/run_inference.py" -i "${RESOLVED_DATASET_DIR}" -o "${OUTPUT_DIR}" -d "${DATASET_NAME}" -s "${SPLIT}" --dino-chunk-size 1000 --max-dense-points 800000 --use-cnos-detection)
  if [ -n "${SCENE_ID}" ]; then CMD+=(--scene-id "${SCENE_ID}"); fi
  if [ -n "${IM_ID}" ]; then CMD+=(--im-id "${IM_ID}"); fi
  if [ -n "${OBJ_ID}" ]; then CMD+=(--obj-id "${OBJ_ID}"); fi
  if [ -n "${INSTANCE_ID}" ]; then CMD+=(--instance-id "${INSTANCE_ID}"); fi
  if [ -n "${SAMPLE_INDEX}" ]; then CMD+=(--sample-index "${SAMPLE_INDEX}"); fi
  if [ -n "${REFINEMENT_DEBUG_DIR}" ]; then CMD+=(--refinement-debug-dir "${REFINEMENT_DEBUG_DIR}"); fi
  if [ -n "${REFINEMENT_GEO_WEIGHT}" ]; then CMD+=(--refinement-geo-weight "${REFINEMENT_GEO_WEIGHT}"); fi
  if [ -n "${REFINEMENT_EPSILON_Z}" ]; then CMD+=(--refinement-epsilon-z "${REFINEMENT_EPSILON_Z}"); fi
  if [ -n "${REFINEMENT_EPSILON_XY}" ]; then CMD+=(--refinement-epsilon-xy "${REFINEMENT_EPSILON_XY}"); fi
  if [ -n "${REFINEMENT_SILHOUETTE_WEIGHT}" ]; then CMD+=(--refinement-silhouette-weight "${REFINEMENT_SILHOUETTE_WEIGHT}"); fi
  if [ -n "${REFINEMENT_ROTATION_PARAMETRIZATION}" ]; then CMD+=(--refinement-rotation-parametrization "${REFINEMENT_ROTATION_PARAMETRIZATION}"); fi
  if [ -n "${REFINEMENT_EPSILON_ROTATION_DEG}" ]; then CMD+=(--refinement-epsilon-rotation-deg "${REFINEMENT_EPSILON_ROTATION_DEG}"); fi
  "${CMD[@]}"
  NOVASPLAT_EXIT_CODE=$?
  if [ ${NOVASPLAT_EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] Novasplat inference failed with exit code ${NOVASPLAT_EXIT_CODE}."
    exit ${NOVASPLAT_EXIT_CODE}
  fi
fi

# Run evaluation w/o refinement
RESULT_CSV="novasplat_${DATASET_NAME}-${SPLIT}.csv"
RESULT_CSV_PATH="${OUTPUT_DIR}/${RESULT_CSV}"
TARGETS_FILENAME="${SPLIT}_targets_bop24.json"

if [ ! -f "${RESULT_CSV_PATH}" ]; then
  echo "[ERROR] Expected inference result CSV not found: ${RESULT_CSV_PATH}"
  echo "[ERROR] Novasplat inference likely failed earlier or wrote to a different output directory."
  exit 1
fi

if [ ! -f "${RESOLVED_DATASET_DIR}/${TARGETS_FILENAME}" ]; then
  echo "[ERROR] Expected BOP targets file not found: ${RESOLVED_DATASET_DIR}/${TARGETS_FILENAME}"
  exit 1
fi

if [ "${SPLIT}" == "val" ]; then
  python thirdparty/bop_toolkit/scripts/eval_bop24_pose.py \
    --datasets_path "${DATASETS_PATH}" \
    --result_filenames "${RESULT_CSV}" \
    --results_path "${OUTPUT_DIR}" \
    --eval_path ${OUTPUT_DIR}/eval \
    --targets_filename "${TARGETS_FILENAME}" \
    --num_workers 24 \
    --use_gpu \
    --device cuda:0
fi

# Run evaluation with refinement
RESULT_CSV="novasplat_${DATASET_NAME}-${SPLIT}_refined.csv"
RESULT_CSV_PATH="${OUTPUT_DIR}/${RESULT_CSV}"
TARGETS_FILENAME="${SPLIT}_targets_bop24.json"

if [ ! -f "${RESULT_CSV_PATH}" ]; then
  echo "[ERROR] Expected inference result CSV not found: ${RESULT_CSV_PATH}"
  echo "[ERROR] Novasplat inference likely failed earlier or wrote to a different output directory."
  exit 1
fi

if [ ! -f "${RESOLVED_DATASET_DIR}/${TARGETS_FILENAME}" ]; then
  echo "[ERROR] Expected BOP targets file not found: ${RESOLVED_DATASET_DIR}/${TARGETS_FILENAME}"
  exit 1
fi

if [ "${SPLIT}" == "val" ]; then
  python thirdparty/bop_toolkit/scripts/eval_bop24_pose.py \
    --datasets_path "${DATASETS_PATH}" \
    --result_filenames "${RESULT_CSV}" \
    --results_path "${OUTPUT_DIR}" \
    --eval_path ${OUTPUT_DIR}/eval \
    --targets_filename "${TARGETS_FILENAME}" \
    --num_workers 24 \
    --use_gpu \
    --device cuda:0
fi

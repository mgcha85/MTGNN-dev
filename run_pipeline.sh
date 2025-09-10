#!/usr/bin/env bash
set -euo pipefail

# === Config ===
export TZ=Asia/Seoul
export NET_ROOT="${NET_ROOT:-/opt/mtgnn}"          # 외부에서 NET_ROOT 주입 가능
PYTHON_BIN="${PYTHON_BIN:-python}"                 # 가상환경 python 경로를 넣어도 됨
LOG_DIR="${NET_ROOT}/logs"
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
mkdir -p "${LOG_DIR}"

# === Logging helper ===
log() { echo "[$(date +'%F %T')][${RUN_ID}] $*"; }

# === Lock (동시 실행 방지) ===
LOCK_FILE="${NET_ROOT}/.pipeline.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  log "Another run is in progress. Exiting."
  exit 1
fi

log "Start pipeline (NET_ROOT=${NET_ROOT})"

# 1) Smoothing
log "Step 1: smoothing.py"
${PYTHON_BIN} smoothing.py \
  >> "${LOG_DIR}/smoothing_${RUN_ID}.log" 2>&1

# 2) Train
log "Step 2: train_new.py"
${PYTHON_BIN} train_new.py \
  --data "${NET_ROOT}/data/sm_data.txt" \
  --save "${NET_ROOT}/model/Bayesian/model.safetensors" \
  --hp_save "${NET_ROOT}/model/Bayesian/hp.txt" \
  >> "${LOG_DIR}/train_${RUN_ID}.log" 2>&1

# 3) Forecast
log "Step 3: forecast_new.py"
${PYTHON_BIN} forecast_new.py \
  --P 10 --out_len 36 \
  >> "${LOG_DIR}/forecast_${RUN_ID}.log" 2>&1

log "Pipeline finished successfully."

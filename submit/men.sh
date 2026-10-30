#!/bin/bash

USER_ID="${USER_ID:?Set USER_ID before submitting a job}"
USER_NAME="${USER_NAME:?Set USER_NAME before submitting a job}"
ACCESS_ID="${ACCESS_ID:-}"
ACCESS_KEY="${ACCESS_KEY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROLEAGENT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}:${PYTHONPATH:-}"

# OSS 与 NAS 配置
ENDPOINT="${OSS_ENDPOINT:?Set OSS_ENDPOINT before submitting a job}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:-}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:-}"
OSS_BUCKET="${OSS_BUCKET:?Set OSS_BUCKET before submitting a job}"
NAS_ENDPOINT="${NAS_ENDPOINT:?Set NAS_ENDPOINT before submitting a job}"

cd "${SCRIPT_DIR}"
workerCount="1"
gpu_per_worker="8"
cluster_file="./cluster_${gpu_per_worker}.json"
timestamp=$(date "+%Y%m%d%H%M%S")

# Parse arguments
ENTRY_SCRIPT=""
JOB_NAME="train"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--entry)
            ENTRY_SCRIPT="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -e <entry_script> -j <job_name>"
            exit 1
            ;;
    esac
done

if [ -z "$ENTRY_SCRIPT" ]; then
    echo "Error: entry script is required. Use -e <path>"
    exit 1
fi

echo "Submitting job: $JOB_NAME"
echo "Entry script: $ENTRY_SCRIPT"

../ai-hub-cli train mdl \
  --name=ehg \
  --job_name="${JOB_NAME}" \
  --entry="bash ${SCRIPT_DIR}/sen.sh --env-path ${ENTRY_SCRIPT}" \
  --file.cluster_file="${cluster_file}" \
  --worker_count="${workerCount}" \
  --algo_name="${AI_HUB_ALGO:?Set AI_HUB_ALGO before submitting a job}" \
  --token="${AI_HUB_TOKEN:?Set AI_HUB_TOKEN before submitting a job}" \
  --namespace="${AI_HUB_NAMESPACE:?Set AI_HUB_NAMESPACE before submitting a job}" \
  --user_params="" \
  --script="" \
  --custom_docker_image="${CUSTOM_DOCKER_IMAGE:?Set CUSTOM_DOCKER_IMAGE before submitting a job}" \
  --env="OMP_NUM_THREADS=${gpu_per_worker},GPUS=${gpu_per_worker},timestamp=${timestamp}" \
  --ignore "outputs/*,storage/*" \
  --nas_file_system_id=9b4854ab18-jyj17.cn-zhangjiakou.nas.aliyuncs.com \
  --queue="${AI_HUB_QUEUE:?Set AI_HUB_QUEUE before submitting a job}"

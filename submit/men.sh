#!/bin/bash

USER_ID="${USER_ID:-528316}"
USER_NAME="${USER_NAME:-wangxucong.wxc}"
ACCESS_ID="${ACCESS_ID:-}"
ACCESS_KEY="${ACCESS_KEY:-}"

export PYTHONPATH=/mnt/workspace/wxc/roleagent:$PYTHONPATH

# OSS 与 NAS 配置
ENDPOINT=cn-shanghai.oss.aliyuncs.com
OSS_ACCESS_ID="${OSS_ACCESS_ID:-}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:-}"
OSS_BUCKET=ml-aigc-multimodal
NAS_ENDPOINT=9b4854ab18-jyj17.cn-zhangjiakou.nas.aliyuncs.com

cd /mnt/workspace/wxc/roleagent/submit
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
  --name=test \
  --job_name="${JOB_NAME}" \
  --entry="bash /mnt/workspace/wxc/roleagent/submit/sen.sh --env-path ${ENTRY_SCRIPT}" \
  --file.cluster_file="${cluster_file}" \
  --worker_count="${workerCount}" \
  --algo_name=pytorch260 \
  --token=974cf14b557151a0d8f5ad72055dbb92 \
  --namespace=aigc_mlvl \
  --user_params="" \
  --script="" \
  --custom_docker_image=reg.docker.alibaba-inc.com/mdl/pytorch220-cu121-jupyterlab-ide:20241107153334 \
  --env="OMP_NUM_THREADS=${gpu_per_worker},GPUS=${gpu_per_worker},timestamp=${timestamp}" \
  --ignore "outputs/*,storage/*" \
  --nas_file_system_id=9b4854ab18-jyj17.cn-zhangjiakou.nas.aliyuncs.com \
  --queue=demand_specific_na175_h20

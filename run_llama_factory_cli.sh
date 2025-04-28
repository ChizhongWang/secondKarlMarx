#!/bin/bash
# 使用LLaMA Factory的官方CLI命令进行训练

# 默认使用所有可用GPU
NUM_GPUS=${1:-"all"}

# 设置环境变量
if [ "$NUM_GPUS" = "all" ]; then
    # 使用所有可用GPU
    unset CUDA_VISIBLE_DEVICES
    echo "使用所有可用GPU"
else
    # 构建GPU ID列表 (0,1,2,...,NUM_GPUS-1)
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    echo "使用GPU: $GPU_LIST"
fi

export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export FORCE_TORCHRUN=1

# 首先准备数据集
echo "准备数据集..."
python prepare_dataset.py

# 使用官方CLI命令
echo "开始训练..."
llamafactory-cli train qwen_lora_sft.yaml

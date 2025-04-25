#!/bin/bash
# 分布式训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 6张GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # 启用InfiniBand（如果有）
export NCCL_IB_HCA=mlx5_0:1  # 根据您的InfiniBand设备调整
export NCCL_SOCKET_IFNAME=eth0  # 根据您的网络接口调整
export NCCL_P2P_LEVEL=NVL  # 优先使用NVLink（如果有）

# 设置Hugging Face认证令牌（如果需要）
# export HUGGING_FACE_HUB_TOKEN=your_token_here

# 设置DeepSpeed环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NODE_RANK=0

# 获取GPU数量
NUM_GPUS=6  # 固定为6张GPU

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 启动分布式训练
deepspeed --num_gpus=$NUM_GPUS train.py \
    --deepspeed configs/ds_config.json \
    --gradient_checkpointing True

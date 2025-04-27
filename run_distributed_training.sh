#!/bin/bash
# 最小可行性实验 - 分布式训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用所有4张GPU
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据您的网络接口调整

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# 设置DeepSpeed环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NODE_RANK=0

# 获取GPU数量
NUM_GPUS=4  # 使用所有4张GPU

# 启动分布式训练
deepspeed --num_gpus=$NUM_GPUS \
    --num_nodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --deepspeed configs/ds_config.json

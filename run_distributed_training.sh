#!/bin/bash
# 最小可行性实验 - 分布式训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU
export NCCL_DEBUG=INFO
# 修改NCCL网络配置
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环接口
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=1  # 禁用P2P通信

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置DeepSpeed环境变量
export MASTER_ADDR=127.0.0.1  # 使用本地回环IP地址
export MASTER_PORT=29500
export NODE_RANK=0

# 获取GPU数量
NUM_GPUS=4  # 使用4张GPU

# 启动分布式训练
deepspeed --num_gpus=$NUM_GPUS \
    --num_nodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --deepspeed configs/ds_config.json

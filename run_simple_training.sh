#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用前两个GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用PyTorch的分布式启动器
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py

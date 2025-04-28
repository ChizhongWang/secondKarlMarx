"""
主训练脚本 - 用于启动分布式训练
"""

import os
import sys
import json
import argparse
import logging
from training.trainer import train
import torch
import deepspeed.launcher
import deepspeed

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train secondKarlMarx model")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        help="DeepSpeed configuration file path",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        help="DeepSpeed configuration file path (alternative to --deepspeed)",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=None,
        help="ZeRO optimization stage",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置环境变量
    if args.local_rank != -1:
        logger.info(f"Running in distributed mode with local_rank: {args.local_rank}")
        # 初始化PyTorch分布式
        if not torch.distributed.is_initialized():
            # 显式设置当前设备，避免NCCL设备映射问题
            torch.cuda.set_device(args.local_rank)
            # 使用环境变量初始化，避免显式指定backend
            torch.distributed.init_process_group(init_method="env://")
            logger.info(f"Initialized process group for rank {torch.distributed.get_rank()}")
    
    # 处理DeepSpeed配置
    deepspeed_config = None
    if args.deepspeed:
        logger.info(f"Loading DeepSpeed config from: {args.deepspeed}")
        with open(args.deepspeed, 'r') as f:
            deepspeed_config = json.load(f)
    elif args.deepspeed_config:
        logger.info(f"Loading DeepSpeed config from: {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    
    # 执行训练
    try:
        final_model_path = train(
            local_rank=args.local_rank,
            deepspeed_config=deepspeed_config,
            zero_stage=args.zero_stage
        )
        logger.info(f"Training completed successfully! Model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config.json")
    args = parser.parse_args()
    
    # 初始化分布式训练
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    
    # 加载DeepSpeed配置
    deepspeed_config = None
    if args.deepspeed:
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    
    # 启动训练
    train(local_rank=args.local_rank, deepspeed_config=deepspeed_config)

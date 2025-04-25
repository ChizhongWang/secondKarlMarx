"""
主训练脚本 - 用于启动分布式训练
"""

import os
import sys
import argparse
import logging
from training.trainer import train

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置环境变量
    if args.local_rank != -1:
        logger.info(f"Running in distributed mode with local_rank: {args.local_rank}")
    
    # 执行训练
    try:
        final_model_path = train()
        logger.info(f"Training completed successfully! Model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()

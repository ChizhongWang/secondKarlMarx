"""
启动MCP服务器脚本 - 用于在云服务器上启动MCP服务
"""

import os
import sys
import logging
import argparse
from mcp.server import karlmarx

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="启动secondKarlMarx MCP服务器")
    parser.add_argument("--model_path", type=str, default="./results/final_model", help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    args = parser.parse_args()
    
    # 更新配置
    from configs.training_config import MCP_CONFIG
    MCP_CONFIG["model_path"] = args.model_path
    MCP_CONFIG["host"] = args.host
    MCP_CONFIG["port"] = args.port
    
    # 启动服务器
    logger.info(f"Starting MCP server for secondKarlMarx on {args.host}:{args.port}")
    logger.info(f"Using model from {args.model_path}")
    
    # 运行MCP服务
    karlmarx.run(transport='stdio')

if __name__ == "__main__":
    main()

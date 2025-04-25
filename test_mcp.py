"""
MCP连接测试脚本 - 用于验证MCP服务是否正常工作
"""

import os
import sys
import logging
import argparse
from mcp.client import Client

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_mcp_connection(server_name="secondKarlMarx"):
    """测试MCP连接"""
    try:
        # 初始化MCP客户端
        client = Client()
        
        # 列出可用服务器
        servers = client.list_servers()
        logger.info(f"可用的MCP服务器: {servers}")
        
        # 检查目标服务器是否可用
        if server_name not in servers:
            logger.error(f"服务器 '{server_name}' 不可用")
            return False
        
        # 测试简单对话
        logger.info(f"向 {server_name} 发送测试消息...")
        response = getattr(client, server_name).chat("你是谁？请简短介绍一下自己。")
        logger.info(f"收到回复: {response}")
        
        logger.info("MCP连接测试成功!")
        return True
    
    except Exception as e:
        logger.error(f"MCP连接测试失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="测试MCP连接")
    parser.add_argument("--server", type=str, default="secondKarlMarx", help="MCP服务器名称")
    args = parser.parse_args()
    
    success = test_mcp_connection(args.server)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的MCP命令行客户端 - 用于在本地连接到远程Qwen2.5-7B-Instruct微调模型
"""

import os
import sys
import logging
import argparse
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MCPClient:
    """简单的MCP命令行客户端"""
    
    def __init__(self):
        """初始化客户端"""
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_server(self, server_host: str, server_port: int):
        """连接到MCP服务器
        
        Args:
            server_host: 服务器主机地址
            server_port: 服务器端口
        """
        # 使用SSH连接到远程服务器并启动MCP服务器
        server_params = StdioServerParameters(
            command="ssh",
            args=[f"{server_host}", "-p", f"{server_port}", "cd /home/featurize/secondKarlMarx/mcp && python server.py"],
            env=None
        )
        
        print(f"正在连接到服务器 {server_host}:{server_port}...")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print(f"已连接到服务器，可用工具: {[tool.name for tool in tools]}")
        
        # 获取模型信息
        try:
            model_info = await self.get_model_info()
            print(f"模型信息: {json.dumps(model_info, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
    
    async def get_model_info(self):
        """获取模型信息"""
        if not self.session:
            return {"model_name": "未连接", "adapter": "未连接"}
        
        try:
            result = await self.session.call_tool("get_model_info", {})
            return json.loads(result.content[0])
        except Exception as e:
            logger.error(f"获取模型信息时出错: {str(e)}")
            return {"model_name": "未知", "adapter": "未知"}
    
    async def chat(self, message):
        """与模型对话"""
        if not self.session:
            return "未连接到服务器"
        
        try:
            result = await self.session.call_tool("chat", {"message": message})
            return result.content[0]
        except Exception as e:
            logger.error(f"对话时出错: {str(e)}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    async def clear_history(self):
        """清除对话历史"""
        if not self.session:
            return "未连接到服务器"
        
        try:
            result = await self.session.call_tool("clear_history", {})
            return "对话历史已清除"
        except Exception as e:
            logger.error(f"清除历史时出错: {str(e)}")
            return f"清除历史时出错: {str(e)}"
    
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP客户端已启动!")
        print("输入您的问题，或输入 'quit' 退出，输入 'clear' 清除对话历史。")
        
        while True:
            try:
                query = input("\n问题: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                if query.lower() == 'clear':
                    result = await self.clear_history()
                    print(result)
                    continue
                
                if not query:
                    continue
                
                print("模型思考中...")
                response = await self.chat(query)
                print(f"\n回答: {response}")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调模型 MCP命令行客户端")
    parser.add_argument("--host", type=str, default="workspace.featurize.cn", help="服务器主机地址")
    parser.add_argument("--port", type=str, default="58260", help="SSH服务器端口")
    args = parser.parse_args()
    
    client = MCPClient()
    try:
        await client.connect_to_server(args.host, args.port)
        await client.chat_loop()
    except Exception as e:
        logger.error(f"错误: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

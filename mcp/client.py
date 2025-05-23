#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP客户端 - 用于在本地连接到远程Qwen2.5-7B-Instruct微调模型
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
import gradio as gr

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SecondKarlMarxClient:
    """Qwen2.5-7B-Instruct微调模型 MCP客户端"""
    
    def __init__(self):
        """初始化客户端"""
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model_info = {"model_name": "未连接", "adapter": "未连接"}
        self.chat_history = []
    
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
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")
        
        # 获取模型信息
        try:
            self.model_info = await self.get_model_info()
            logger.info(f"Model info: {self.model_info}")
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
    
    async def get_model_info(self):
        """获取模型信息"""
        if not self.session:
            return {"model_name": "未连接", "adapter": "未连接"}
        
        try:
            result = await self.session.call_tool("get_model_info", {})
            return json.loads(result.content[0])
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"model_name": "未知", "adapter": "未知"}
    
    async def chat(self, message):
        """与模型对话"""
        if not self.session:
            return "未连接到服务器"
        
        try:
            result = await self.session.call_tool("chat", {"message": message})
            return result.content[0]
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    async def clear_history(self):
        """清除对话历史"""
        if not self.session:
            return "未连接到服务器"
        
        try:
            result = await self.session.call_tool("clear_history", {})
            self.chat_history = []
            return "对话历史已清除"
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return f"清除历史时出错: {str(e)}"
    
    async def create_ui(self):
        """创建Gradio UI界面"""
        with gr.Blocks(title="Qwen2.5微调模型") as demo:
            gr.Markdown("# Qwen2.5-7B-Instruct 微调模型")
            gr.Markdown("通过MCP连接到云服务器上的微调模型")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 模型信息")
                    gr.Markdown(f"**基础模型**: {self.model_info.get('model_name', '未知')}")
                    gr.Markdown(f"**适配器**: {self.model_info.get('adapter', '未知')}")
                    gr.Markdown(f"**微调类型**: {self.model_info.get('finetuning_type', 'LoRA')}")
                
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="对话历史")
                    msg = gr.Textbox(label="输入您的问题", lines=2)
                    with gr.Row():
                        submit = gr.Button("发送")
                        clear = gr.Button("清除对话历史")
            
            async def user(user_message, history):
                return "", history + [[user_message, None]]
            
            async def bot(history):
                user_message = history[-1][0]
                bot_response = await self.chat(user_message)
                history[-1][1] = bot_response
                return history
            
            async def clear_chat_history():
                await self.clear_history()
                return None
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(clear_chat_history, None, chatbot)
            
        return demo
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调模型 MCP客户端")
    parser.add_argument("--host", type=str, default="workspace.featurize.cn", help="服务器主机地址")
    parser.add_argument("--port", type=str, default="58260", help="SSH服务器端口")
    args = parser.parse_args()
    
    client = SecondKarlMarxClient()
    try:
        await client.connect_to_server(args.host, args.port)
        demo = await client.create_ui()
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

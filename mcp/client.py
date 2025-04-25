"""
MCP客户端 - 用于在本地连接到远程secondKarlMarx模型
"""

import os
import sys
import logging
import argparse
from mcp.client import Client
import gradio as gr

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SecondKarlMarxClient:
    """secondKarlMarx MCP客户端"""
    
    def __init__(self, server_config_path=None):
        """
        初始化客户端
        
        Args:
            server_config_path: MCP服务器配置文件路径
        """
        # 初始化MCP客户端
        self.client = Client()
        self.chat_history = []
        
        # 检查连接
        try:
            self.check_connection()
            logger.info("Successfully connected to secondKarlMarx MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            print(f"错误: 无法连接到secondKarlMarx服务器。请确保服务器正在运行。\n详细信息: {str(e)}")
    
    def check_connection(self):
        """检查与MCP服务器的连接"""
        # 尝试访问服务器
        servers = self.client.list_servers()
        if "secondKarlMarx" not in servers:
            raise ConnectionError("secondKarlMarx服务器未找到。可用的服务器: " + ", ".join(servers))
    
    def chat(self, message):
        """
        与secondKarlMarx模型对话
        
        Args:
            message: 用户消息
            
        Returns:
            模型回复
        """
        try:
            response = self.client.secondKarlMarx.chat(message)
            return response
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def clear_history(self):
        """清除对话历史"""
        try:
            response = self.client.secondKarlMarx.clear_history()
            self.chat_history = []
            return "对话历史已清除"
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return f"清除历史时出错: {str(e)}"
    
    def create_ui(self):
        """创建Gradio UI界面"""
        with gr.Blocks(title="secondKarlMarx") as demo:
            gr.Markdown("# secondKarlMarx - 马克思主义理论助手")
            gr.Markdown("通过MCP连接到云服务器上的secondKarlMarx模型")
            
            chatbot = gr.Chatbot(label="对话历史")
            msg = gr.Textbox(label="输入您的问题", lines=2)
            clear = gr.Button("清除对话历史")
            
            def user(user_message, history):
                return "", history + [[user_message, None]]
            
            def bot(history):
                user_message = history[-1][0]
                bot_response = self.chat(user_message)
                history[-1][1] = bot_response
                return history
            
            def clear_chat_history():
                self.clear_history()
                return None
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(clear_chat_history, None, chatbot)
            
        return demo

def main():
    parser = argparse.ArgumentParser(description="secondKarlMarx MCP客户端")
    parser.add_argument("--config", type=str, help="MCP服务器配置文件路径")
    args = parser.parse_args()
    
    # 创建客户端
    client = SecondKarlMarxClient(args.config)
    
    # 启动UI
    demo = client.create_ui()
    demo.launch(share=True)

if __name__ == "__main__":
    main()

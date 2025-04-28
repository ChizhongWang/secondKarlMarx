"""
MCP客户端 - 用于在本地连接到远程Qwen2.5-7B-Instruct微调模型
"""

import os
import sys
import logging
import argparse
import json
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
    """Qwen2.5-7B-Instruct微调模型 MCP客户端"""
    
    def __init__(self, server_host=None, server_port=None):
        """
        初始化客户端
        
        Args:
            server_host: MCP服务器主机地址
            server_port: MCP服务器端口
        """
        # 初始化MCP客户端
        self.client = Client()
        self.chat_history = []
        
        # 更新MCP配置
        if server_host or server_port:
            self.update_mcp_config(server_host, server_port)
        
        # 检查连接
        try:
            self.check_connection()
            logger.info("Successfully connected to secondKarlMarx MCP server")
            
            # 获取模型信息
            self.model_info = self.get_model_info()
            logger.info(f"Model info: {self.model_info}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            print(f"错误: 无法连接到secondKarlMarx服务器。请确保服务器正在运行。\n详细信息: {str(e)}")
            self.model_info = {"model_name": "未连接", "adapter": "未连接"}
    
    def update_mcp_config(self, server_host, server_port):
        """更新MCP配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "mcp_config.json")
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            if server_host:
                config["mcpServers"]["secondKarlMarx"]["host"] = server_host
            
            if server_port:
                config["mcpServers"]["secondKarlMarx"]["port"] = int(server_port)
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Updated MCP config: host={server_host}, port={server_port}")
        except Exception as e:
            logger.error(f"Failed to update MCP config: {str(e)}")
    
    def check_connection(self):
        """检查与MCP服务器的连接"""
        # 尝试访问服务器
        servers = self.client.list_servers()
        if "secondKarlMarx" not in servers:
            raise ConnectionError("secondKarlMarx服务器未找到。可用的服务器: " + ", ".join(servers))
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            return self.client.secondKarlMarx.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"model_name": "未知", "adapter": "未知"}
    
    def chat(self, message):
        """
        与Qwen2.5-7B-Instruct微调模型对话
        
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
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(clear_chat_history, None, chatbot)
            
        return demo

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调模型 MCP客户端")
    parser.add_argument("--host", type=str, help="MCP服务器主机地址")
    parser.add_argument("--port", type=str, help="MCP服务器端口")
    args = parser.parse_args()
    
    # 创建客户端
    client = SecondKarlMarxClient(args.host, args.port)
    
    # 启动UI
    demo = client.create_ui()
    demo.launch(share=True)

if __name__ == "__main__":
    main()

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
import gradio as gr

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def update_mcp_config(config_path, server_host=None, server_port=None):
    """更新MCP配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if server_host:
            config["mcpServers"]["secondKarlMarx"]["host"] = server_host
        
        if server_port:
            config["mcpServers"]["secondKarlMarx"]["port"] = int(server_port)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated MCP config: host={server_host}, port={server_port}")
    except Exception as e:
        logger.error(f"Failed to update MCP config: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调模型 MCP客户端")
    parser.add_argument("--host", type=str, help="MCP服务器主机地址")
    parser.add_argument("--port", type=str, help="MCP服务器端口")
    args = parser.parse_args()
    
    # 更新MCP配置
    config_path = os.path.join(os.path.dirname(__file__), "mcp_config.json")
    if args.host or args.port:
        update_mcp_config(config_path, args.host, args.port)
    
    # 启动客户端
    try:
        # 导入MCP客户端
        import mcp
        client = mcp.Client()
        
        # 检查可用的服务器
        servers = client.list_servers()
        if "secondKarlMarx" not in servers:
            print(f"错误: secondKarlMarx服务器未找到。可用的服务器: {', '.join(servers)}")
            return
        
        print("成功连接到secondKarlMarx MCP服务器")
        
        # 获取模型信息
        try:
            model_info = client.secondKarlMarx.get_model_info()
            print(f"模型信息: {json.dumps(model_info, indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"获取模型信息失败: {str(e)}")
            model_info = {"model_name": "未知", "adapter": "未知", "finetuning_type": "未知", "template": "未知"}
        
        # 创建Gradio界面
        with gr.Blocks(title="Qwen2.5微调模型") as demo:
            gr.Markdown("# Qwen2.5-7B-Instruct 微调模型")
            gr.Markdown("通过MCP连接到云服务器上的微调模型")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 模型信息")
                    gr.Markdown(f"**基础模型**: {model_info.get('model_name', '未知')}")
                    gr.Markdown(f"**适配器**: {model_info.get('adapter', '未知')}")
                    gr.Markdown(f"**微调类型**: {model_info.get('finetuning_type', 'LoRA')}")
                
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
                bot_response = client.secondKarlMarx.chat(user_message)
                history[-1][1] = bot_response
                return history
            
            def clear_chat_history():
                client.secondKarlMarx.clear_history()
                return None
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(clear_chat_history, None, chatbot)
        
        # 启动Gradio界面
        demo.launch(share=True)
        
    except ImportError:
        print("错误: 未找到MCP客户端库。请使用 'pip install mcp-client' 安装。")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()

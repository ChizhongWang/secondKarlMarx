"""
MCP服务器 - 为训练好的模型提供MCP接口
"""

import os
import sys
import logging
import torch
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
import http.client
import json

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.training_config import MCP_CONFIG
from model.model_loader import load_model_for_inference

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 创建MCP服务
karlmarx = FastMCP(MCP_CONFIG["server_name"])

# 加载模型
model, tokenizer = load_model_for_inference(MCP_CONFIG["model_path"])

# 消息历史
messages = []

@karlmarx.tool()
async def chat(query: str) -> str:
    """
    与secondKarlMarx模型进行对话
    
    Args:
        query (str): 用户的问题或指令
    
    Returns:
        str: 模型的回答
    """
    global messages
    
    try:
        # 添加用户消息
        messages.append({"role": "user", "content": query})
        
        # 准备输入
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码回答
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # 添加助手消息
        messages.append({"role": "assistant", "content": response})
        
        # 如果消息历史太长，保留最近的10条
        if len(messages) > 20:
            messages = messages[-20:]
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return f"抱歉，处理您的请求时出现错误: {str(e)}"

@karlmarx.tool()
async def clear_history() -> str:
    """
    清除对话历史
    
    Returns:
        str: 操作结果
    """
    global messages
    messages = []
    return "对话历史已清除"

if __name__ == "__main__":
    # 初始化并运行服务器
    logger.info(f"Starting MCP server for secondKarlMarx on {MCP_CONFIG['host']}:{MCP_CONFIG['port']}")
    karlmarx.run(transport='stdio')

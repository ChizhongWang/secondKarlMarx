"""
简单的MCP客户端 - 用于在本地连接到远程Qwen2.5-7B-Instruct微调模型
"""

import os
import sys
import logging
import argparse
import json
import requests

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SimpleMCPClient:
    """简单的MCP客户端"""
    
    def __init__(self, host, port):
        """初始化客户端"""
        self.base_url = f"http://{host}:{port}"
        self.history = []
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/tools/get_model_info",
                json={}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"model_name": "未知", "adapter": "未知"}
    
    def chat(self, message):
        """与模型对话"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/tools/chat",
                json={"message": message}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def clear_history(self):
        """清除对话历史"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/tools/clear_history",
                json={}
            )
            response.raise_for_status()
            self.history = []
            return "对话历史已清除"
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return f"清除历史时出错: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调模型 MCP客户端")
    parser.add_argument("--host", type=str, default="localhost", help="MCP服务器主机地址")
    parser.add_argument("--port", type=str, default="8000", help="MCP服务器端口")
    args = parser.parse_args()
    
    # 初始化客户端
    client = SimpleMCPClient(args.host, args.port)
    
    try:
        # 获取模型信息
        model_info = client.get_model_info()
        print(f"模型信息: {json.dumps(model_info, indent=2, ensure_ascii=False)}")
        
        # 开始聊天
        print("\n=== 开始与Qwen2.5-7B-Instruct微调模型对话 ===")
        print("输入 'exit' 或 'quit' 退出, 输入 'clear' 清除对话历史")
        
        while True:
            try:
                user_input = input("\n用户: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    print("再见!")
                    break
                
                if user_input.lower() == "clear":
                    result = client.clear_history()
                    print(result)
                    continue
                
                if not user_input.strip():
                    continue
                
                print("模型思考中...")
                response = client.chat(user_input)
                print(f"\n模型: {response}")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {str(e)}")
    
    except Exception as e:
        print(f"连接到MCP服务器时出错: {str(e)}")

if __name__ == "__main__":
    main()

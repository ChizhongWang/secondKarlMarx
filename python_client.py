#!/usr/bin/env python
# 用于从本地访问云服务器上的模型API

import requests
import json
import argparse

def chat_with_model(server_ip, port, prompt, history=None):
    """
    与云服务器上的模型进行对话
    
    Args:
        server_ip: 云服务器IP地址
        port: API服务端口
        prompt: 用户输入的提示
        history: 对话历史
    
    Returns:
        模型的回复
    """
    url = f"http://{server_ip}:{port}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 构建消息历史
    messages = []
    if history:
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
    
    # 添加当前提示
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": "Qwen2.5-7B-Instruct-LoRA",
        "messages": messages
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()  # 如果响应状态码不是200，将引发异常
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"错误: 无法连接到服务器 - {str(e)}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"错误: 解析响应时出错 - {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="与微调后的Qwen模型对话")
    parser.add_argument("--ip", required=True, help="云服务器IP地址")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口")
    
    args = parser.parse_args()
    
    print(f"连接到 {args.ip}:{args.port} 上的模型...")
    print("输入 'exit' 或 'quit' 结束对话")
    print("-" * 50)
    
    history = []
    
    while True:
        user_input = input("\n用户: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("再见!")
            break
        
        print("\n正在生成回复...")
        response = chat_with_model(args.ip, args.port, user_input, history)
        
        print(f"\n助手: {response}")
        
        # 更新历史
        history.append((user_input, response))

if __name__ == "__main__":
    main()

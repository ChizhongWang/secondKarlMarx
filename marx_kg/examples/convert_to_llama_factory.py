"""
将工具调用示例转换为LLaMA Factory格式
用于将工具调用示例转换为LLaMA Factory可用的训练数据格式
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_examples(input_file: str) -> List[Dict[str, Any]]:
    """加载工具调用示例"""
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    return examples

def convert_to_llama_factory(examples: List[Dict[str, Any]], output_format: str = "alpaca") -> List[Dict[str, Any]]:
    """
    将工具调用示例转换为LLaMA Factory格式
    
    Args:
        examples: 工具调用示例列表
        output_format: 输出格式，支持"alpaca"和"sharegpt"
        
    Returns:
        转换后的示例列表
    """
    converted_examples = []
    
    for example in examples:
        conversations = example["conversations"]
        
        if output_format == "alpaca":
            # Alpaca格式: {"instruction": "...", "input": "", "output": "..."}
            # 我们将用户问题作为instruction，助手的最终回答作为output
            user_message = next((msg["content"] for msg in conversations if msg["role"] == "user"), "")
            assistant_final = next((msg["content"] for msg in conversations if msg["role"] == "assistant" and "tool_calls" not in msg), "")
            
            converted_examples.append({
                "instruction": user_message,
                "input": "",
                "output": assistant_final
            })
            
        elif output_format == "sharegpt":
            # ShareGPT格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
            # 我们保留完整的对话，包括工具调用
            sharegpt_conversations = []
            
            for msg in conversations:
                if msg["role"] == "user":
                    sharegpt_conversations.append({
                        "from": "human",
                        "value": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    # 如果有工具调用，将其添加到消息中
                    content = msg["content"]
                    if "tool_calls" in msg:
                        tool_calls_text = "\n\n<tool_calls>\n"
                        for tool_call in msg["tool_calls"]:
                            tool_calls_text += f"调用工具: {tool_call['name']}\n"
                            tool_calls_text += f"参数: {json.dumps(tool_call['arguments'], ensure_ascii=False)}\n"
                        tool_calls_text += "</tool_calls>"
                        content += tool_calls_text
                    
                    sharegpt_conversations.append({
                        "from": "gpt",
                        "value": content
                    })
                elif msg["role"] == "tool":
                    # 工具响应作为系统消息
                    sharegpt_conversations.append({
                        "from": "system",
                        "value": f"工具 {msg['tool_call_id']} 返回结果: {msg['content']}"
                    })
            
            converted_examples.append({
                "conversations": sharegpt_conversations
            })
        
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    return converted_examples

def save_examples(examples: List[Dict[str, Any]], output_file: str):
    """保存转换后的示例"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将工具调用示例转换为LLaMA Factory格式")
    parser.add_argument("--input", type=str, default="tool_calling_examples.json", help="输入文件路径")
    parser.add_argument("--output", type=str, default="llama_factory_examples.json", help="输出文件路径")
    parser.add_argument("--format", type=str, choices=["alpaca", "sharegpt"], default="sharegpt", help="输出格式")
    args = parser.parse_args()
    
    # 确保输入文件路径是绝对路径
    input_file = args.input
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)
    
    # 确保输出文件路径是绝对路径
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), output_file)
    
    print(f"加载工具调用示例: {input_file}")
    examples = load_examples(input_file)
    
    print(f"转换为LLaMA Factory {args.format}格式")
    converted_examples = convert_to_llama_factory(examples, args.format)
    
    print(f"保存转换后的示例: {output_file}")
    save_examples(converted_examples, output_file)
    
    print(f"转换完成! 共转换 {len(converted_examples)} 个示例")

if __name__ == "__main__":
    main()

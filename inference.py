"""
推理脚本 - 用于加载和运行微调后的模型
"""

import os
import sys
import logging
from model.model_loader import load_model_for_inference
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # 设置模型路径
    model_path = "./outputs"
    
    # 加载模型和分词器
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_inference(model_path)
    
    # 设置模型为评估模式
    model.eval()
    
    # 示例对话
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的问题（输入'quit'退出）: ")
            if user_input.lower() == 'quit':
                break
            
            # 构建提示
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成回复
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):]  # 移除提示部分
            
            # 打印回复
            print("\n助手:", response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            continue

if __name__ == "__main__":
    main()
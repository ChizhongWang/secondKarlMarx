"""
数据处理工具 - 用于加载和预处理训练数据
"""

import logging
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_sft_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_length: int = 2048,
    text_column: str = "text",
    use_auth_token: bool = False,
    data_dir: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    """
    加载SFT数据集并进行预处理
    
    Args:
        dataset_name: Hugging Face数据集名称
        tokenizer: 用于分词的tokenizer
        max_seq_length: 最大序列长度
        text_column: 数据集中文本所在的列名
        use_auth_token: 是否使用Hugging Face认证令牌
        data_dir: 本地数据目录（如果不从Hugging Face加载）
        split: 数据集分割
        streaming: 是否使用流式加载
        
    Returns:
        处理后的数据集
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # 加载数据集
    if data_dir:
        dataset = load_dataset(
            dataset_name,
            data_dir=data_dir,
            split=split,
            streaming=streaming,
            use_auth_token=use_auth_token
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            use_auth_token=use_auth_token
        )
    
    logger.info(f"Dataset loaded with {len(dataset) if not streaming else 'streaming'} examples")
    
    # 检查数据集格式
    sample = dataset[0] if not streaming else next(iter(dataset))
    logger.info(f"Sample data: {sample}")
    
    # 确定数据集格式
    if "instruction" in sample and "output" in sample:
        logger.info("Detected instruction-output format")
        format_type = "instruction-output"
    elif "messages" in sample:
        logger.info("Detected messages format")
        format_type = "messages"
    elif text_column in sample:
        logger.info(f"Using {text_column} column")
        format_type = "text"
    else:
        raise ValueError(f"Unsupported dataset format. Sample: {sample}")
    
    # 根据不同格式处理数据
    def preprocess_function(examples):
        if format_type == "instruction-output":
            texts = []
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples.get("input", [""]*len(examples["instruction"])),
                examples["output"]
            ):
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
        
        elif format_type == "messages":
            texts = []
            for msg_list in examples["messages"]:
                text = ""
                for msg in msg_list:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        text += f"### System:\n{content}\n\n"
                    elif role == "user":
                        text += f"### User:\n{content}\n\n"
                    elif role == "assistant":
                        text += f"### Assistant:\n{content}\n\n"
                texts.append(text)
        
        elif format_type == "text":
            texts = examples[text_column]
        
        # 分词
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 创建标签（用于因果语言模型训练）
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 应用预处理
    if streaming:
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    else:
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing and formatting dataset",
        )
    
    logger.info(f"Processed dataset: {processed_dataset}")
    return processed_dataset

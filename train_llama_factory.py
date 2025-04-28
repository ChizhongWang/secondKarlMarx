"""
使用LLaMA Factory进行模型微调
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, set_seed
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft import train_model
from llmtuner.extras.misc import torch_gc

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型参数
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B",
        metadata={"help": "基础模型的路径或名称"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速分词器"}
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "是否调整词表大小"}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "量化位数"}
    )

@dataclass
class DataArguments:
    """
    数据参数
    """
    dataset: str = field(
        default="custom",
        metadata={"help": "数据集名称"}
    )
    dataset_dir: Optional[str] = field(
        default="./training",
        metadata={"help": "数据集目录"}
    )
    template: str = field(
        default="qwen",
        metadata={"help": "模板名称"}
    )
    cutoff_len: int = field(
        default=8192,
        metadata={"help": "截断长度"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    训练参数
    """
    finetuning_type: str = field(
        default="lora",
        metadata={"help": "微调类型：lora, qlora, full"}
    )
    lora_target: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "LoRA目标模块"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    output_dir: str = field(
        default="./outputs_llama_factory",
        metadata={"help": "输出目录"}
    )

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 加载模型和分词器
    logger.info("加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(
        model_args=model_args,
        finetuning_args=training_args,
        is_trainable=True
    )
    
    # 训练模型
    logger.info("开始训练...")
    train_model(
        model=model,
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    
    # 清理内存
    torch_gc()

if __name__ == "__main__":
    main()
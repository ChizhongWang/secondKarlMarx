#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
马克思文本向量化工具

该脚本用于将马克思恩格斯全集的文本块向量化，并存储到向量数据库中。
使用DMXAPI的text-embedding-3-large模型生成向量。

作者: 王驰中
日期: 2025-05-09
"""

import os
import json
import time
import httpx
import numpy as np
import lancedb
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vectorize.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarxVectorizer:
    def __init__(self, api_key, api_base="https://www.dmxapi.cn/v1", batch_size=20, max_workers=5):
        """初始化向量化器
        
        Args:
            api_key: DMXAPI的API密钥
            api_base: API基础URL
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        self.api_key = api_key
        self.api_base = api_base
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.client = httpx.Client(timeout=60.0)
        self.vector_dim = 3072  # text-embedding-3-large的维度
        
    def generate_embedding(self, text, max_retries=3):
        """为单个文本生成嵌入向量
        
        Args:
            text: 输入文本
            max_retries: 最大重试次数
            
        Returns:
            嵌入向量或None（如果失败）
        """
        # 截断文本，避免超出API限制
        if len(text) > 8000:
            text = text[:4000] + text[-4000:]
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "text-embedding-3-large",
            "input": text
        }
        
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    return embedding
                else:
                    logger.warning(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {response.status_code} {response.text}")
                    time.sleep(2 ** attempt)  # 指数退避
            except Exception as e:
                logger.error(f"API调用异常 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(2 ** attempt)
                
        return None
    
    def batch_generate_embeddings(self, texts):
        """批量生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            embeddings = list(executor.map(self.generate_embedding, texts))
        return embeddings
    
    def process_files(self, input_dir, output_dir=None, db_path=None, checkpoint_file="checkpoint.json"):
        """处理所有文本文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选）
            db_path: 向量数据库路径（可选）
            checkpoint_file: 检查点文件
        """
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 初始化向量数据库
        db = None
        table = None
        if db_path:
            os.makedirs(os.path.dirname(db_path) if not os.path.exists(os.path.dirname(db_path)) else db_path, exist_ok=True)
            db = lancedb.connect(db_path)
            
            # 检查表是否存在
            if "marx_texts" not in db.table_names():
                schema = {
                    "id": "string",
                    "vector": f"float32[{self.vector_dim}]",
                    "text": "string",
                    "volume": "string",
                    "article_number": "string",
                    "title": "string",
                    "chunk_id": "string",
                    "parent_article": "string",
                    "chunk_type": "string"
                }
                table = db.create_table("marx_texts", schema=schema)
                logger.info(f"创建新表: marx_texts")
            else:
                table = db.open_table("marx_texts")
                logger.info(f"打开现有表: marx_texts")
        
        # 加载检查点
        processed_files = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                processed_files = set(json.load(f))
            logger.info(f"从检查点加载已处理文件: {len(processed_files)}个")
        
        # 获取所有JSON文件
        all_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
        
        logger.info(f"找到 {len(all_files)} 个文本块文件")
        remaining_files = [f for f in all_files if f not in processed_files]
        logger.info(f"需要处理 {len(remaining_files)} 个文件")
        
        # 批量处理
        for i in tqdm(range(0, len(remaining_files), self.batch_size)):
            batch_files = remaining_files[i:i+self.batch_size]
            texts = []
            file_data = []
            
            # 读取文件内容
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        texts.append(chunk['raw_text'])
                        file_data.append((file_path, chunk))
                except Exception as e:
                    logger.error(f"读取文件出错 {file_path}: {str(e)}")
                    processed_files.add(file_path)  # 标记为已处理，避免重复尝试
            
            # 生成向量
            vectors = self.batch_generate_embeddings(texts)
            
            # 处理结果
            db_records = []
            for j, ((file_path, chunk), vector) in enumerate(zip(file_data, vectors)):
                if vector is None:
                    logger.warning(f"文件 {file_path} 向量生成失败，跳过")
                    continue
                    
                metadata = chunk["metadata"]
                chunk_id = f"{metadata.get('volume', 'unknown')}_{metadata.get('article_number', 'unknown')}_{metadata.get('chunk_id', 'unknown')}"
                
                # 保存到文件
                if output_dir:
                    vector_data = {
                        "metadata": metadata,
                        "raw_text": chunk["raw_text"],
                        "vector": vector
                    }
                    
                    base_name = os.path.basename(file_path)
                    output_path = os.path.join(output_dir, base_name)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(vector_data, f, ensure_ascii=False, indent=2)
                
                # 添加到数据库
                if table:
                    db_record = {
                        "id": chunk_id,
                        "vector": vector,
                        "text": chunk["raw_text"],
                        "volume": metadata.get("volume", ""),
                        "article_number": metadata.get("article_number", ""),
                        "title": metadata.get("title", ""),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "parent_article": metadata.get("parent_article", ""),
                        "chunk_type": metadata.get("chunk_type", "")
                    }
                    db_records.append(db_record)
                
                # 标记为已处理
                processed_files.add(file_path)
            
            # 批量添加到数据库
            if table and db_records:
                try:
                    table.add(db_records)
                except Exception as e:
                    logger.error(f"添加到数据库出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 更新检查点
            with open(checkpoint_file, 'w') as f:
                json.dump(list(processed_files), f)
            
            # 小休息，避免API限制
            time.sleep(0.5)
        
        logger.info(f"向量化处理完成，共处理 {len(processed_files)} 个文件")
        
        # 显示数据库统计信息
        if table:
            count = table.count()
            logger.info(f"向量数据库包含 {count} 条记录")

def main():
    parser = argparse.ArgumentParser(description='马克思文本向量化工具')
    parser.add_argument('--input_dir', required=True, help='输入目录，包含JSON文本块')
    parser.add_argument('--output_dir', help='输出目录，保存带向量的JSON文件')
    parser.add_argument('--db_path', help='向量数据库路径')
    parser.add_argument('--api_key', required=True, help='DMXAPI密钥')
    parser.add_argument('--batch_size', type=int, default=20, help='批处理大小')
    parser.add_argument('--max_workers', type=int, default=5, help='最大并行工作线程数')
    parser.add_argument('--checkpoint_file', default="checkpoint.json", help='检查点文件路径')
    
    args = parser.parse_args()
    
    if not args.output_dir and not args.db_path:
        parser.error("必须指定至少一个输出位置: --output_dir 或 --db_path")
    
    vectorizer = MarxVectorizer(
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    vectorizer.process_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_path=args.db_path,
        checkpoint_file=args.checkpoint_file
    )

if __name__ == "__main__":
    main()

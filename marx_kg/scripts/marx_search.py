#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
马克思文本搜索工具

该脚本用于在向量化后的马克思恩格斯全集文本中进行语义搜索，
并支持获取相关文本的上下文和完整文章。

作者: Cascade
日期: 2025-05-09
"""

import os
import json
import time
import httpx
import lancedb
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarxTextSearcher:
    def __init__(self, api_key: str, db_path: str, chunks_dir: str, api_base: str = "https://www.dmxapi.cn/v1"):
        """初始化搜索器
        
        Args:
            api_key: DMXAPI的API密钥
            db_path: 向量数据库路径
            chunks_dir: 原始文本块目录
            api_base: API基础URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self.db_path = db_path
        self.chunks_dir = chunks_dir
        self.client = httpx.Client(timeout=30.0)
        
        # 连接数据库
        self.db = lancedb.connect(db_path)
        if "marx_texts" not in self.db.table_names():
            raise ValueError(f"数据库中不存在marx_texts表，请先运行向量化脚本")
        self.table = self.db.open_table("marx_texts")
        logger.info(f"已连接到向量数据库，表: marx_texts")
        
        # 构建文本块索引
        self.chunk_index = self._build_chunk_index()
        logger.info(f"已构建文本块索引，包含 {len(self.chunk_index)} 个条目")
    
    def _build_chunk_index(self) -> Dict[str, str]:
        """构建文本块ID到文件路径的索引"""
        chunk_index = {}
        for root, _, files in os.walk(self.chunks_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            chunk = json.load(f)
                        metadata = chunk["metadata"]
                        chunk_id = f"{metadata.get('volume', 'unknown')}_{metadata.get('article_number', 'unknown')}_{metadata.get('chunk_id', 'unknown')}"
                        chunk_index[chunk_id] = file_path
                    except Exception as e:
                        logger.warning(f"构建索引时读取文件出错 {file_path}: {str(e)}")
        return chunk_index
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """为查询文本生成嵌入向量
        
        Args:
            query_text: 查询文本
            
        Returns:
            查询向量
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "text-embedding-3-large",
            "input": query_text
        }
        
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
                logger.error(f"生成查询向量失败: {response.status_code} {response.text}")
                return None
        except Exception as e:
            logger.error(f"生成查询向量出错: {str(e)}")
            return None
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索与查询文本相似的文本块
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似文本块列表
        """
        # 生成查询向量
        query_vector = self.generate_query_embedding(query_text)
        if query_vector is None:
            logger.error("无法生成查询向量，搜索失败")
            return []
        
        # 执行向量搜索
        try:
            results = self.table.search(query_vector).limit(top_k).to_pandas()
            logger.info(f"搜索完成，找到 {len(results)} 个相关文本块")
            
            # 转换为字典列表
            search_results = []
            for _, row in results.iterrows():
                search_results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "volume": row["volume"],
                    "article_number": row["article_number"],
                    "title": row["title"],
                    "chunk_id": row["chunk_id"],
                    "parent_article": row["parent_article"],
                    "chunk_type": row["chunk_type"],
                    "score": float(row["_distance"])  # 相似度分数
                })
            
            return search_results
        except Exception as e:
            logger.error(f"搜索出错: {str(e)}")
            return []
    
    def get_context(self, chunk_id: str, context_size: int = 1) -> List[Dict[str, Any]]:
        """获取指定文本块的上下文
        
        Args:
            chunk_id: 文本块ID (volume_article_number_chunk_id格式)
            context_size: 上下文大小（前后各多少个块）
            
        Returns:
            上下文文本块列表
        """
        try:
            # 解析chunk_id
            parts = chunk_id.split('_')
            if len(parts) < 3:
                logger.error(f"无效的chunk_id格式: {chunk_id}")
                return []
            
            volume = parts[0]
            article_number = parts[1]
            chunk_num = int(parts[2])
            
            # 查找相同文章的所有块
            context_chunks = []
            for i in range(max(0, chunk_num - context_size), chunk_num + context_size + 1):
                context_id = f"{volume}_{article_number}_{i}"
                if context_id in self.chunk_index:
                    file_path = self.chunk_index[context_id]
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                    
                    context_chunks.append({
                        "id": context_id,
                        "text": chunk["raw_text"],
                        "metadata": chunk["metadata"],
                        "is_target": (i == chunk_num)  # 标记是否为目标块
                    })
            
            return context_chunks
        except Exception as e:
            logger.error(f"获取上下文出错: {str(e)}")
            return []
    
    def get_full_article(self, volume: str, article_number: str) -> List[Dict[str, Any]]:
        """获取完整文章的所有文本块
        
        Args:
            volume: 卷号
            article_number: 文章编号
            
        Returns:
            文章的所有文本块
        """
        try:
            article_chunks = []
            article_prefix = f"{volume}_{article_number}_"
            
            # 查找所有属于该文章的块
            for chunk_id, file_path in self.chunk_index.items():
                if chunk_id.startswith(article_prefix):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                    
                    article_chunks.append({
                        "id": chunk_id,
                        "text": chunk["raw_text"],
                        "metadata": chunk["metadata"]
                    })
            
            # 按chunk_id排序
            article_chunks.sort(key=lambda x: int(x["metadata"]["chunk_id"]))
            
            return article_chunks
        except Exception as e:
            logger.error(f"获取完整文章出错: {str(e)}")
            return []
    
    def search_with_context(self, query_text: str, top_k: int = 5, context_size: int = 1) -> Dict[str, Any]:
        """搜索并返回带上下文的结果
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            context_size: 上下文大小
            
        Returns:
            搜索结果和上下文
        """
        # 执行基本搜索
        search_results = self.search(query_text, top_k)
        
        # 为每个结果获取上下文
        results_with_context = []
        for result in search_results:
            context = self.get_context(result["id"], context_size)
            results_with_context.append({
                "result": result,
                "context": context
            })
        
        return {
            "query": query_text,
            "results": results_with_context
        }

def format_search_results(results: Dict[str, Any], show_context: bool = True) -> str:
    """格式化搜索结果为可读文本
    
    Args:
        results: 搜索结果
        show_context: 是否显示上下文
        
    Returns:
        格式化的文本
    """
    output = f"查询: {results['query']}\n\n"
    output += f"找到 {len(results['results'])} 个相关文本块:\n\n"
    
    for i, item in enumerate(results['results']):
        result = item['result']
        output += f"--- 结果 {i+1} (相似度: {result['score']:.4f}) ---\n"
        output += f"标题: {result['title']}\n"
        output += f"出处: 《{result['parent_article']}》\n"
        output += f"位置: 第{result['volume']}卷, 文章{result['article_number']}, 块{result['chunk_id']}\n\n"
        
        # 显示文本摘要
        text = result['text']
        if len(text) > 300:
            output += f"{text[:300]}...\n\n"
        else:
            output += f"{text}\n\n"
        
        # 显示上下文
        if show_context and 'context' in item and item['context']:
            output += "上下文:\n"
            for ctx in item['context']:
                if ctx['is_target']:
                    output += "【当前块】\n"
                else:
                    output += f"【相邻块 {ctx['metadata']['chunk_id']}】\n"
                
                # 显示上下文摘要
                ctx_text = ctx['text']
                if len(ctx_text) > 200:
                    output += f"{ctx_text[:200]}...\n\n"
                else:
                    output += f"{ctx_text}\n\n"
        
        output += "-" * 50 + "\n\n"
    
    return output

def main():
    parser = argparse.ArgumentParser(description='马克思文本搜索工具')
    parser.add_argument('--query', required=True, help='搜索查询')
    parser.add_argument('--db_path', required=True, help='向量数据库路径')
    parser.add_argument('--chunks_dir', required=True, help='原始文本块目录')
    parser.add_argument('--api_key', required=True, help='DMXAPI密钥')
    parser.add_argument('--top_k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--context_size', type=int, default=1, help='上下文大小')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    searcher = MarxTextSearcher(
        api_key=args.api_key,
        db_path=args.db_path,
        chunks_dir=args.chunks_dir
    )
    
    results = searcher.search_with_context(
        query_text=args.query,
        top_k=args.top_k,
        context_size=args.context_size
    )
    
    formatted_results = format_search_results(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        print(f"搜索结果已保存到 {args.output}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    main()

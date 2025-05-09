"""
马克思恩格斯知识图谱增强LLM脚本
将微调的Qwen模型与GraphRAG知识图谱结合
"""
import os
import sys
import json
import math
import glob
import logging
import requests
import asyncio
import tempfile
import pickle
import pandas as pd
import networkx as nx
import tiktoken
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Any, Tuple, NamedTuple

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.local_search_config import LocalSearchConfig
from graphrag.data_model.text_unit import TextUnit
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.query.factory import get_local_search_engine
from graphrag.language_model.manager import ModelManager

# 配置日志
import io
import sys

# 解决Windows控制台的编码问题
if sys.platform == 'win32':
    # 强制使用UTF-8编码
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 自定义StreamHandler，确保正确处理Unicode
class EncodingStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_enhanced_llm.log", encoding='utf-8'),
        EncodingStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义TextChunk类用于存储原始文本块
class TextChunk(NamedTuple):
    """TextChunk类用于存储原始文本块"""
    id: str
    text: str
    volume: str
    article_number: str
    title: str
    chunk_id: str
    embedding: Optional[List[float]] = None

class KGEnhancedLLM:
    """知识图谱增强的LLM"""
    
    def __init__(self, llm_api_url="http://localhost:8000/v1/chat/completions", use_kg=True):
        """初始化
        
        Args:
            llm_api_url: LLM API的URL
            use_kg: 是否使用知识图谱增强
        """
        self.llm_api_url = llm_api_url
        self.use_kg = use_kg
        self.graph = None
        self.search_engine = None
        self.use_embeddings = False
        self.use_graphrag = True  # 默认使用GraphRAG，只有在初始化失败时才会设置为False
        self.text_chunks = []
        self.embedding_model = None
        
        if use_kg:
            try:
                # 初始化知识图谱查询引擎
                self.initialize_kg_query()
                
                # 初始化文本检索引擎
                self.initialize_text_retrieval()
                
                # 检查GraphRAG是否成功初始化
                if self.use_graphrag:
                    if hasattr(self, 'search_engine') and self.search_engine is not None:
                        logger.info("GraphRAG查询引擎和文本检索引擎初始化成功")
                    else:
                        logger.warning("GraphRAG查询引擎初始化失败，将使用NetworkX进行查询")
                        self.use_graphrag = False
                else:
                    logger.info("将使用NetworkX进行查询")
            except Exception as e:
                logger.error(f"初始化知识图谱查询引擎时出错: {str(e)}", exc_info=True)
                self.use_kg = False
                self.use_graphrag = False
                
    def query_kg(self, query_text):
        """查询知识图谱和相关文本
        
        Args:
            query_text: 查询文本
            
        Returns:
            str: 知识图谱和文本检索结果
        """
        if not self.use_kg:
            return None
            
        try:
            # 处理查询文本
            query_text = query_text.encode('utf-8', errors='ignore').decode('utf-8')
            logger.info(f"查询知识图谱和相关文本: {query_text}")
            
            # 初始化结果变量
            kg_result = ""
            text_result = ""
            
            # 记录use_graphrag的值
            logger.info(f"use_graphrag的当前值: {self.use_graphrag}")
            logger.info(f"search_engine是否存在: {hasattr(self, 'search_engine') and self.search_engine is not None}")
            
            # 使用GraphRAG查询知识图谱
            if self.use_graphrag and hasattr(self, 'search_engine') and self.search_engine is not None:
                try:
                    # 为查询生成嵌入向量
                    embedding_config = LanguageModelConfig(
                        type="openai_embedding",
                        model="text-embedding-ada-002",
                        encoding_model="cl100k_base",
                        api_key=os.environ.get("DMX_API_KEY"),
                        api_base="https://www.dmxapi.cn/v1"
                    )
                    
                    # 获取嵌入模型
                    embedding_model = ModelManager().get_or_create_embedding_model(
                        "query_embedding_model",
                        "openai_embedding",
                        config=embedding_config
                    )
                    
                    # 保存嵌入模型供文本检索使用
                    self.embedding_model = embedding_model
                    
                    # 使用GraphRAG原生的查询方法
                    # 1. 修改上下文构建器参数
                    context_builder_params = {
                        "top_k_entities": 10,         # 减少返回的实体数量，提高相关性
                        "top_k_relationships": 10,    # 适当减少关系数量，提高相关性
                        "text_unit_prop": 0.0,        # 不使用文本单元
                        "community_prop": 0.0,        # 不使用社区报告
                        "include_entity_rank": True,  # 包含实体排名
                        "include_relationship_weight": True,  # 包含关系权重
                        "similarity_threshold": 0.6,  # 添加相似度阈值，只返回相关性较高的实体
                        "max_hops": 1                # 限制关系查询的深度，减少无关实体
                    }
                    
                    # 2. 记录搜索引擎的类型和属性
                    logger.info(f"搜索引擎类型: {type(self.search_engine)}")
                    logger.info(f"搜索引擎属性: {dir(self.search_engine)}")
                    
                    # 3. 设置搜索引擎参数
                    try:
                        self.search_engine.context_builder_params.update(context_builder_params)
                        logger.info("成功更新上下文构建器参数")
                    except Exception as e:
                        logger.error(f"更新上下文构建器参数时出错: {str(e)}", exc_info=True)
                    
                    # 4. 使用search方法查询知识图谱
                    try:
                        logger.info(f"开始执行GraphRAG查询: {query_text}")
                        result = asyncio.run(self.search_engine.search(
                            query=query_text
                        ))
                        logger.info(f"GraphRAG查询执行完成，结果类型: {type(result)}")
                        
                        if result:
                            logger.info(f"GraphRAG结果属性: {dir(result)}")
                        
                        if result and hasattr(result, 'context_text'):
                            logger.info(f"GraphRAG查询结果: {result.context_text[:200]}...")
                            kg_result = result.context_text
                        else:
                            logger.warning(f"GraphRAG查询返回的结果为空或格式不正确: {result}")
                            # 回退到NetworkX查询
                            raise Exception("回退到NetworkX查询: 结果为空或格式不正确")
                    except Exception as e:
                        logger.error(f"执行GraphRAG search方法时出错: {str(e)}", exc_info=True)
                        # 如果GraphRAG查询失败，回退到NetworkX查询
                        raise Exception(f"回退到NetworkX查询: GraphRAG search失败 - {str(e)}")
                except Exception as e:
                    logger.error(f"使用GraphRAG查询时出错: {str(e)}", exc_info=True)
                    # 如果GraphRAG查询失败，回退到NetworkX查询
                    raise Exception(f"回退到NetworkX查询: {str(e)}")
            
            # 回退到NetworkX查询
            if not kg_result and hasattr(self, 'graph') and self.graph is not None:
                # 使用NetworkX直接查询
                # 使用更精确的关键词匹配和相关性计算
                query_text_lower = query_text.lower()
                
                # 将查询分解为关键词
                query_keywords = [word for word in query_text_lower.split() if len(word) > 1]
                
                # 为特定查询添加额外的关键词
                if '马克思恩格斯全集' in query_text:
                    query_keywords.extend(['全集', '著作', '文集', '卷', '原文'])
                    
                if '第一卷' in query_text or '第1卷' in query_text:
                    query_keywords.extend(['第一卷', '第1卷', '卷一', '第一册'])
                    
                if '原文' in query_text:
                    query_keywords.extend(['原著', '文本', '文献'])
                
                # 计算每个节点的相关性分数
                node_scores = {}
                for node, data in self.graph.nodes(data=True):
                    node_text = f"{node} {data.get('description', '')}".lower()
                    if pd.isna(node_text):
                        node_text = str(node).lower()
                    
                    # 计算关键词匹配分数 - 增加权重
                    keyword_score = sum(3 for keyword in query_keywords if keyword in node_text)
                    
                    # 计算完整知识图谱查询匹配分数
                    full_query_score = 5 if query_text_lower in node_text else 0
                    
                    # 计算节点中关键实体匹配分数
                    entity_score = 0
                    important_entities = ['马克思', '恩格斯', '全集', '第一卷', '著作', '文集']
                    for entity in important_entities:
                        if entity.lower() in node_text:
                            entity_score += 3
                    
                    # 特殊处理：如果查询与马克思恩格斯全集相关
                    if '马克思恩格斯全集' in query_text and ('全集' in node_text or '马克思恩格斯全集' in node_text):
                        entity_score += 10
                    
                    # 总分数
                    total_score = keyword_score + full_query_score + entity_score
                    
                    if total_score > 0:
                        node_scores[node] = total_score
                
                # 按分数降序排序并选取前10个节点
                # 将节点和分数组成元组列表
                top_nodes = [(node, node_scores[node]) for node in sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)[:10]]
                
                if not top_nodes:
                    kg_result = "知识图谱查询结果:\n\n未找到与查询相关的实体。\n\n未找到与查询相关的关系。"
                else:
                    # 查找这些节点之间的关系，以及与查询相关的其他关系
                    related_edges = []
                    
                    # 首先查找这些节点之间的直接关系
                    # 从 top_nodes 中提取节点ID
                    related_node_ids = [node for node, _ in top_nodes]
                    
                    for i, node1 in enumerate(related_node_ids):
                        for node2 in related_node_ids[i+1:]:
                            if self.graph.has_edge(node1, node2):
                                edge_data = self.graph.get_edge_data(node1, node2)
                                related_edges.append((node1, node2, edge_data))
                            elif self.graph.has_edge(node2, node1):
                                edge_data = self.graph.get_edge_data(node2, node1)
                                related_edges.append((node2, node1, edge_data))
                    
                    # 如果直接关系很少，扩展查找一跳关系
                    if len(related_edges) < 5:
                        # 查找与高分节点相连的其他节点
                        # 使用前3个高分节点
                        high_score_nodes = [node for node, _ in top_nodes[:3]]
                        for node in high_score_nodes:
                            # 查找节点的直接邻居
                            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
                            for neighbor in neighbors:
                                if neighbor not in related_node_ids:
                                    # 计算邻居节点的相关性
                                    neighbor_data = self.graph.nodes[neighbor]
                                    neighbor_text = f"{neighbor} {neighbor_data.get('description', '')}".lower()
                                    
                                    # 检查是否与查询相关
                                    if any(keyword in neighbor_text for keyword in query_keywords):
                                        # 添加这个关系
                                        if self.graph.has_edge(node, neighbor):
                                            edge_data = self.graph.get_edge_data(node, neighbor)
                                            related_edges.append((node, neighbor, edge_data))
                                        elif self.graph.has_edge(neighbor, node):
                                            edge_data = self.graph.get_edge_data(neighbor, node)
                                            related_edges.append((neighbor, node, edge_data))
                
                # 计算每个关系的相关性分数
                edge_scores = {}
                for source, target, data in related_edges:
                    edge_text = f"{source} {target} {data.get('description', '')}".lower()
                    
                    # 计算关键词匹配分数
                    keyword_score = sum(1 for keyword in query_keywords if keyword in edge_text)
                    
                    # 计算与相关节点的连接分数
                    connection_score = 0
                    if source in node_scores:
                        connection_score += node_scores[source] * 0.5
                    if target in node_scores:
                        connection_score += node_scores[target] * 0.5
                    
                    # 总分数
                    total_score = keyword_score + connection_score
                    
                    if total_score > 0:
                        edge_id = f"{source}-{target}"
                        edge_scores[edge_id] = (source, target, data, total_score)
                
                # 按分数降序排序并选取前10个关系
                top_edges = sorted(edge_scores.values(), key=lambda x: x[3], reverse=True)[:10]
                
                # 构建结果字符串 - 更好的格式化
                result = "### 相关实体与概念\n\n"
                
                # 检查top_nodes的结构
                logger.info(f"top_nodes类型: {type(top_nodes)}, 长度: {len(top_nodes)}")
                if top_nodes and len(top_nodes) > 0:
                    logger.info(f"top_nodes第一个元素类型: {type(top_nodes[0])}, 值: {top_nodes[0]}")
                
                for item in top_nodes:
                    # 适应不同的数据结构
                    if isinstance(item, tuple) and len(item) == 2:
                        node, score = item
                    elif isinstance(item, str):
                        node = item
                        score = 1  # 默认分数
                    else:
                        logger.warning(f"未知的top_nodes元素类型: {type(item)}, 值: {item}")
                        continue
                        
                    node_data = self.graph.nodes[node]
                    description = node_data.get('description', '')
                    if pd.isna(description):
                        description = "无描述"
                    result += f"**{node}** (相关度: {score})\n"
                    result += f"描述: {description}\n\n"
                
                result += "### 相关关系\n\n"
                for source, target, data, score in top_edges:
                    description = data.get('description', '')
                    if pd.isna(description):
                        description = f"从 {source} 到 {target} 的关系"
                    result += f"**{source}** → **{target}** (相关度: {score:.1f})\n"
                    result += f"关系描述: {description}\n\n"
                
                # 添加查询总结
                result += "### 知识图谱查询总结\n\n"
                
                # 针对特定查询添加更多信息
                if '马克思恩格斯全集' in query_text:
                    if '第一卷' in query_text or '第1卷' in query_text:
                        result += "查询与《马克思恩格斯全集》第一卷相关。根据知识图谱信息，《马克思恩格斯全集》第一卷包含马克思早期著作，"
                        result += "主要是1833年至1843年间的作品，包括马克思的博士论文和早期政论文章。\n\n"
                    else:
                        result += "查询与《马克思恩格斯全集》相关。《马克思恩格斯全集》是马克思和恩格斯著作的完整集合，"
                        result += "包含他们的所有著作、手稿、信件和笔记。\n\n"
                
                if '原文' in query_text:
                    result += "查询涉及原文内容。马克思恩格斯的原始著作主要是用德语写作的，部分著作也有法语和英语版本。\n\n"
                
                logger.info(f"NetworkX查询结果: {result[:500]}...")
                kg_result = result       # 检索相关文本
            relevant_texts = self.retrieve_relevant_texts(query_text, top_k=3)
            if relevant_texts:
                text_result = "\n\n-----原文引用-----\n"
                for chunk in relevant_texts:
                    text_result += f"\n《马克思恩格斯全集》第{chunk.volume}卷《{chunk.title}》:\n"
                    # 截取部分文本，避免过长
                    text_snippet = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
                    text_result += f"{text_snippet}\n"
            
            # 组合知识图谱和文本检索结果
            combined_result = kg_result
            if text_result:
                combined_result += text_result
            
            return combined_result
        except Exception as e:
            logger.error(f"查询知识图谱时出错: {str(e)}", exc_info=True)
            return f"查询知识图谱时出错: {str(e)}"
                
    def initialize_text_retrieval(self):
        """初始化文本检索引擎"""
        try:
            # 查找所有文本块文件
            chunks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_chunks")
            logger.info(f"加载文本块目录: {chunks_dir}")
            
            # 检查目录是否存在
            if not os.path.exists(chunks_dir):
                logger.warning(f"文本块目录 {chunks_dir} 不存在，跳过文本检索初始化")
                return
            
            # 加载所有JSON文件
            json_files = glob.glob(os.path.join(chunks_dir, "*.json"))
            logger.info(f"找到 {len(json_files)} 个文本块文件")
            
            # 处理每个文件
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        
                    # 创建TextChunk对象
                    text_chunk = TextChunk(
                        id=os.path.basename(file_path),
                        text=chunk_data.get("raw_text", ""),
                        volume=chunk_data.get("metadata", {}).get("volume", ""),
                        article_number=chunk_data.get("metadata", {}).get("article_number", ""),
                        title=chunk_data.get("metadata", {}).get("title", ""),
                        chunk_id=chunk_data.get("metadata", {}).get("chunk_id", ""),
                        embedding=None  # 初始化为空
                    )
                    self.text_chunks.append(text_chunk)
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {str(e)}", exc_info=True)
            
            logger.info(f"成功加载 {len(self.text_chunks)} 个文本块")
            
            # 如果有嵌入模型，为文本块生成嵌入向量
            if self.embedding_model is not None:
                logger.info("开始为文本块生成嵌入向量...")
                for i, chunk in enumerate(self.text_chunks):
                    try:
                        # 只使用文本的前1000个字符生成嵌入向量，以提高效率
                        text_for_embedding = chunk.text[:1000]
                        embedding = self.embedding_model.embed(text_for_embedding)
                        
                        # 创建新的TextChunk对象并替换原来的
                        self.text_chunks[i] = chunk._replace(embedding=embedding)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"已完成 {i+1}/{len(self.text_chunks)} 个文本块的嵌入向量生成")
                    except Exception as e:
                        logger.error(f"为文本块 {chunk.id} 生成嵌入向量时出错: {str(e)}", exc_info=True)
                
                logger.info("文本块嵌入向量生成完成")
        except Exception as e:
            logger.error(f"初始化文本检索引擎时出错: {str(e)}", exc_info=True)
    
    def retrieve_relevant_texts(self, query_text, top_k=3):
        """检索与查询相关的文本
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数量
            
        Returns:
            List[TextChunk]: 相关文本块列表
        """
        if not self.text_chunks:
            logger.warning("没有可用的文本块进行检索")
            return []
        
        # 如果有嵌入模型和文本嵌入向量，使用向量相似度检索
        if self.embedding_model is not None and any(chunk.embedding is not None for chunk in self.text_chunks):
            try:
                # 生成查询的嵌入向量
                query_embedding = self.embedding_model.embed(query_text)
                
                # 计算相似度
                similarities = []
                for chunk in self.text_chunks:
                    if chunk.embedding is not None:
                        # 使用余弦相似度
                        similarity = np.dot(query_embedding, chunk.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                        )
                        similarities.append((chunk, similarity))
                    else:
                        similarities.append((chunk, 0.0))
                
                # 按相似度降序排序
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 返回前 top_k 个结果
                return [chunk for chunk, _ in similarities[:top_k]]
            except Exception as e:
                logger.error(f"使用嵌入向量检索文本时出错: {str(e)}", exc_info=True)
        
        # 如果没有嵌入向量或嵌入检索失败，回退到关键词匹配
        logger.info("使用关键词匹配检索文本")
        query_keywords = [word for word in query_text.lower().split() if len(word) > 1]
        
        # 计算每个文本块的匹配分数
        chunk_scores = []
        for chunk in self.text_chunks:
            # 关键词匹配分数
            keyword_score = sum(2 for keyword in query_keywords if keyword in chunk.text.lower())
            
            # 完整查询匹配分数
            full_query_score = 3 if query_text.lower() in chunk.text.lower() else 0
            
            # 标题匹配分数
            title_score = sum(2 for keyword in query_keywords if keyword in chunk.title.lower())
            
            # 总分数
            total_score = keyword_score + full_query_score + title_score
            
            if total_score > 0:
                chunk_scores.append((chunk, total_score))
        
        # 按分数降序排序
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 top_k 个结果
        return [chunk for chunk, _ in chunk_scores[:top_k]]
    
    def initialize_kg_query(self):
        """初始化知识图谱查询引擎"""
        try:
            # 加载配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "graphrag_config.yaml")
            logger.info(f"加载配置文件: {config_path}")
            
            # 创建临时目录用于LanceDB
            temp_dir = tempfile.mkdtemp()
            logger.info(f"创建LanceDB数据库目录: {temp_dir}")
            
            # 加载实体和关系数据
            entities_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kg", "entities.csv")
            relationships_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kg", "relationships.csv")
            logger.info(f"加载实体数据: {entities_path}")
            logger.info(f"加载关系数据: {relationships_path}")
            
            # 读取CSV文件
            entities_df = pd.read_csv(entities_path)
            relationships_df = pd.read_csv(relationships_path)
            
            # 创建NetworkX图
            self.graph = nx.DiGraph()
            
            # 添加实体节点
            for _, row in entities_df.iterrows():
                entity_id = row['title']  # 使用title作为实体ID
                self.graph.add_node(entity_id, **row.to_dict())
            
            # 添加关系边
            for _, row in relationships_df.iterrows():
                source = row['source']
                target = row['target']
                self.graph.add_edge(source, target, **row.to_dict())
            
            # 创建实体和关系对象
            entities = []
            for _, row in entities_df.iterrows():
                entity = Entity(
                    id=row['title'],  # 使用title作为实体ID
                    short_id=row['title'],  # 使用title作为short_id
                    title=row['title'],
                    type=row['type'],
                    description=row.get('description', '')
                )
                entities.append(entity)
            
            relationships = []
            for _, row in relationships_df.iterrows():
                relationship = Relationship(
                    id=f"{row['source']}-{row['target']}",  # 使用source-target作为关系ID
                    short_id=f"{row['source']}-{row['target']}",  # 使用source-target作为short_id
                    source=row['source'],
                    target=row['target'],
                    weight=float(row.get('strength', 0.5)),
                    description=row.get('description', '')
                )
                relationships.append(relationship)
            
            # 获取API密钥
            api_key = os.environ.get("DMX_API_KEY")
            if not api_key:
                logger.warning("DMX_API_KEY环境变量未设置，将使用简单的关键词匹配进行查询")
                # 使用简单的关键词匹配作为备选
                self.use_embeddings = False
                return
            
            # 创建LanceDB向量存储
            vector_store = LanceDBVectorStore(collection_name="descriptions")
            db_uri = os.path.join(temp_dir)
            vector_store.connect(db_uri=db_uri)
            
            # 创建嵌入模型配置
            embedding_config = LanguageModelConfig(
                type="openai_embedding",
                model="text-embedding-ada-002",
                api_base="https://www.dmxapi.cn/v1",
                api_key=api_key
            )
            
            # 使用ModelManager获取嵌入模型
            embedding_model = ModelManager().get_or_create_embedding_model(
                "entity_embedding_model",
                "openai_embedding",
                config=embedding_config
            )
            
            # 为每个实体创建文档并生成嵌入向量
            documents = []
            for entity in entities:
                try:
                    # 检查实体描述是否为空或非字符串
                    if not entity.description or not isinstance(entity.description, str) or len(str(entity.description).strip()) == 0:
                        logger.warning(f"实体 '{entity.id}' 的描述为空或非字符串类型，跳过嵌入向量生成")
                        continue
                        
                    # 为实体描述生成嵌入向量
                    entity_vector = embedding_model.embed(entity.description)
                    
                    # 检查向量中是否有NaN值
                    if any(math.isnan(x) for x in entity_vector):
                        logger.warning(f"实体 '{entity.id}' 的嵌入向量包含NaN值，跳过该实体")
                        continue
                        
                    logger.info(f"为实体 '{entity.id}' 生成嵌入向量成功")
                    
                    doc = VectorStoreDocument(
                        id=entity.id,
                        text=entity.description,
                        vector=entity_vector,  # 直接使用浮点数数组
                        attributes={"entity_id": entity.id}
                    )
                    documents.append(doc)
                except Exception as e:
                    # 如果生成嵌入向量失败，记录错误但继续处理下一个实体
                    logger.error(f"为实体 '{entity.id}' 生成嵌入向量失败: {str(e)}", exc_info=True)
            
            # 加载文档到向量存储
            if documents:
                logger.info(f"加载{len(documents)}个实体描述到向量存储")
                vector_store.load_documents(documents)
                self.use_embeddings = True
            else:
                logger.warning("没有成功生成任何嵌入向量，将使用简单的关键词匹配进行查询")
                self.use_embeddings = False

            # 创建聊天模型配置
            chat_model_config = LanguageModelConfig(
                type="openai_chat",
                model="gpt-3.5-turbo",  # 模型名称不重要，因为我们使用本地API
                encoding_model="cl100k_base",
                api_key="dummy-key",  # 本地API不需要真正的API密钥
                api_base="http://localhost:8000/v1"  # 指向本地模型API
            )

            # 创建本地搜索配置
            local_search_config = LocalSearchConfig(
                text_unit_prop=0.6,
                community_prop=0.4,
                top_k_entities=15,
                top_k_relationships=10,
                embedding_model_id="default_embedding_model",
                chat_model_id="default_chat_model"
            )

            # 创建GraphRagConfig对象
            config = GraphRagConfig(
                models={
                    "default_embedding_model": embedding_config,  # 使用已定义的embedding_config
                    "default_chat_model": chat_model_config  # 使用正确的聊天模型配置
                },
                local_search=local_search_config
            )
            
            # 创建GraphRAG查询引擎
            self.search_engine = get_local_search_engine(
                config=config,
                reports=[],  # 空的报告列表
                text_units=[],  # 空的文本单元列表
                entities=list(entities),  # 将dict_values转换为列表
                relationships=list(relationships),  # 将dict_values转换为列表
                covariates={},  # 空的协变量字典
                response_type="multiple paragraphs",
                description_embedding_store=vector_store,
                callbacks=[QueryCallbacks()],  # 将QueryCallbacks对象包装在一个列表中
                system_prompt="你是一个基于马克思主义理论的助手，请根据提供的上下文回答问题。"
            )
            
            logger.info("GraphRAG查询引擎初始化完成")
            
        except Exception as e:
            logger.error(f"初始化GraphRAG查询引擎时出错: {str(e)}", exc_info=True)
            # 回退到NetworkX查询
            self.use_graphrag = False
            
    def retrieve_relevant_texts(self, query_text, top_k=3):
        """检索与查询相关的文本
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数量
            
        Returns:
            List[TextChunk]: 相关文本块列表
        """
        if not self.text_chunks:
            logger.warning("文本块列表为空，无法进行文本检索")
            return []
            
        # 处理查询文本
        query_text = query_text.encode('utf-8', errors='ignore').decode('utf-8')
        query_text_lower = query_text.lower()
        
        # 将查询分解为关键词
        query_keywords = [word for word in query_text_lower.split() if len(word) > 1]
        
        # 如果有嵌入模型，使用向量相似度查询
        if self.embedding_model and self.use_embeddings:
            try:
                # 为查询生成嵌入向量
                query_embedding = self.embedding_model.embed_query(query_text)
                
                # 计算每个文本块与查询的相似度
                chunk_scores = []
                for chunk in self.text_chunks:
                    if chunk.embedding:
                        # 计算余弦相似度
                        similarity = np.dot(query_embedding, chunk.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                        )
                        chunk_scores.append((chunk, similarity))
                
                # 按相似度降序排序
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 返回前 top_k 个结果
                return [chunk for chunk, _ in chunk_scores[:top_k]]
            except Exception as e:
                logger.error(f"使用向量相似度检索文本时出错: {str(e)}", exc_info=True)
                # 回退到关键词匹配
                pass
        
        # 使用关键词匹配作为备选
        chunk_scores = []
        for chunk in self.text_chunks:
            # 计算关键词匹配分数
            keyword_score = sum(2 for keyword in query_keywords if keyword in chunk.text.lower())
            
            # 计算完整查询匹配分数
            full_query_score = 3 if query_text_lower in chunk.text.lower() else 0
            
            # 标题匹配分数
            title_score = sum(2 for keyword in query_keywords if keyword in chunk.title.lower())
            
            # 总分数
            total_score = keyword_score + full_query_score + title_score
            
            if total_score > 0:
                chunk_scores.append((chunk, total_score))
        
        # 按分数降序排序
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 top_k 个结果
        return [chunk for chunk, _ in chunk_scores[:top_k]]
        
    def initialize_kg_query(self):
        """初始化知识图谱查询引擎"""
        try:
            # 加载配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "graphrag_config.yaml")
            logger.info(f"加载配置文件: {config_path}")
        
            # 创建临时目录用于LanceDB
            temp_dir = tempfile.mkdtemp()
            logger.info(f"创建LanceDB数据库目录: {temp_dir}")
            
            # 加载实体和关系数据
            entities_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kg", "entities.csv")
            relationships_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kg", "relationships.csv")
            logger.info(f"加载实体数据: {entities_path}")
            logger.info(f"加载关系数据: {relationships_path}")
            
            # 读取CSV文件
            entities_df = pd.read_csv(entities_path)
            relationships_df = pd.read_csv(relationships_path)
            
            # 创建NetworkX图
            self.graph = nx.DiGraph()
            
            # 添加实体节点
            for _, row in entities_df.iterrows():
                entity_id = row['title']  # 使用title作为实体ID
                self.graph.add_node(entity_id, **row.to_dict())
            
            # 添加关系边
            for _, row in relationships_df.iterrows():
                source = row['source']
                target = row['target']
                self.graph.add_edge(source, target, **row.to_dict())
            
            # 创建实体和关系对象
            entities = []
            for _, row in entities_df.iterrows():
                entity = Entity(
                    id=row['title'],  # 使用title作为实体ID
                    short_id=row['title'],  # 使用title作为short_id
                    title=row['title'],
                    type=row['type'],
                    description=row.get('description', '')
                )
                entities.append(entity)
            
            relationships = []
            for _, row in relationships_df.iterrows():
                relationship = Relationship(
                    id=f"{row['source']}-{row['target']}",  # 使用source-target作为关系ID
                    short_id=f"{row['source']}-{row['target']}",  # 使用source-target作为short_id
                    source=row['source'],
                    target=row['target'],
                    weight=float(row.get('strength', 0.5)),
                    description=row.get('description', '')
                )
                relationships.append(relationship)
            
            # 获取API密钥
            api_key = os.environ.get("DMX_API_KEY")
            if not api_key:
                logger.warning("DMX_API_KEY环境变量未设置，将使用简单的关键词匹配进行查询")
                # 使用简单的关键词匹配作为备选
                self.use_embeddings = False
                return
            
            # 创建LanceDB向量存储
            vector_store = LanceDBVectorStore(collection_name="descriptions")
            db_uri = os.path.join(temp_dir)
            vector_store.connect(db_uri=db_uri)
            
            # 创建嵌入模型配置
            embedding_config = LanguageModelConfig(
                type="openai_embedding",
                model="text-embedding-ada-002",
                api_base="https://www.dmxapi.cn/v1",
                api_key=api_key
            )
            
            # 使用ModelManager获取嵌入模型
            embedding_model = ModelManager().get_or_create_embedding_model(
                "entity_embedding_model",
                "openai_embedding",
                config=embedding_config
            )
            
            # 为每个实体创建文档并生成嵌入向量
            documents = []
            for entity in entities:
                try:
                    # 检查实体描述是否为空或非字符串
                    if not entity.description or not isinstance(entity.description, str) or len(str(entity.description).strip()) == 0:
                        logger.warning(f"实体 '{entity.id}' 的描述为空或非字符串类型，跳过嵌入向量生成")
                        continue
                    
                        # 为实体描述生成嵌入向量
                    try:
                        entity_vector = embedding_model.embed(entity.description)
                        
                        # 检查向量中是否有NaN值
                        if any(math.isnan(x) for x in entity_vector):
                            logger.warning(f"实体 '{entity.id}' 的嵌入向量包含NaN值，使用默认向量")
                            entity_vector = [0.0] * 1536  # 使用默认向量
                    except Exception as e:
                        logger.warning(f"为实体 '{entity.id}' 生成嵌入向量失败: {str(e)}，使用默认向量")
                        entity_vector = [0.0] * 1536  # 使用默认向量
                    
                    # 创建文档
                    document = VectorStoreDocument(
                        id=entity.id,
                        text=entity.description,
                        vector=entity_vector,
                        attributes={"entity_id": entity.id}
                    )
                    documents.append(document)
                except Exception as e:
                    logger.error(f"处理实体 '{entity.id}' 时出错: {str(e)}", exc_info=True)
                    continue
            
            # 生成嵌入向量并添加到向量存储
            try:
                # 批量生成嵌入向量
                if documents:
                    logger.info(f"为{len(documents)}个实体生成嵌入向量")
                    vector_store.load_documents(documents, embedding_model)
                    logger.info("嵌入向量生成完成并添加到向量存储")
                else:
                    logger.warning("没有有效的实体文档，跳过嵌入向量生成")
            except Exception as e:
                logger.error(f"生成嵌入向量时出错: {str(e)}", exc_info=True)
                self.use_embeddings = False
                
                # 创建聊天模型配置
                chat_model_config = LanguageModelConfig(
                    type="openai_chat",
                    model="gpt-3.5-turbo",
                    api_base="https://www.dmxapi.cn/v1",
                    api_key=api_key
                )
                
                # 创建本地搜索配置
                local_search_config = LocalSearchConfig(
                    similarity_threshold=0.6,
                    top_k_entities=10,
                    top_k_relationships=10,
                    max_hops=1,
                    text_unit_prop=0.0,  # 不使用文本单元
                    community_prop=0.0   # 不使用社区报告
                )
                
                # 创建GraphRagConfig对象
                config = GraphRagConfig(
                    models={
                        "default_embedding_model": embedding_config,
                        "default_chat_model": chat_model_config
                    },
                    local_search=local_search_config
                )
                
                try:
                    # 创建GraphRAG查询引擎
                    logger.info("开始创建GraphRAG查询引擎...")
                    logger.info(f"entities数量: {len(entities)}, relationships数量: {len(relationships)}")
                    
                    # 记录当前的use_graphrag值
                    logger.info(f"GraphRAG初始化前的use_graphrag值: {self.use_graphrag}")
                    
                    try:
                        self.search_engine = get_local_search_engine(
                            config=config,
                            reports=[],
                            text_units=[],
                            entities=list(entities),
                            relationships=list(relationships),
                            covariates={},
                            response_type="multiple paragraphs",
                            description_embedding_store=vector_store,
                            callbacks=[QueryCallbacks()],
                            system_prompt="你是一个基于马克思主义理论的助手，请根据提供的上下文回答问题。"
                        )
                        
                        # 检查search_engine是否成功创建
                        if self.search_engine is not None:
                            logger.info(f"search_engine创建成功，类型: {type(self.search_engine)}")
                            # 设置标志表示我们应该使用GraphRAG
                            self.use_graphrag = True
                            logger.info("GraphRAG查询引擎初始化成功，将使用GraphRAG进行查询")
                        else:
                            logger.error("search_engine创建失败，返回了None")
                            self.use_graphrag = False
                            logger.info("将使用NetworkX进行查询")
                    except Exception as e:
                        logger.error(f"get_local_search_engine调用失败: {str(e)}", exc_info=True)
                        self.use_graphrag = False
                        logger.info("将使用NetworkX进行查询")
                except Exception as e:
                    logger.error(f"GraphRAG查询引擎初始化失败: {str(e)}", exc_info=True)
                    self.use_graphrag = False
                    logger.info("将使用NetworkX进行查询")
            
        except Exception as e:
            logger.error(f"初始化知识图谱查询引擎时出错: {str(e)}", exc_info=True)
            # 回退到NetworkX查询
            self.use_graphrag = False
            
            # 创建NetworkX图
            self.graph = nx.DiGraph()
            
            # 加载实体和关系
            entities_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entities.csv")
            relationships_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "relationships.csv")
            
            entities_df = pd.read_csv(entities_path)
            relationships_df = pd.read_csv(relationships_path)
            
            # 添加节点
            for _, row in entities_df.iterrows():
                self.graph.add_node(row['title'], **row.to_dict())
            
            # 添加边
            for _, row in relationships_df.iterrows():
                source = row['source']
                target = row['target']
                self.graph.add_edge(source, target, **row.to_dict())
            
            logger.info(f"NetworkX图创建完成，共有{self.graph.number_of_nodes()}个节点和{self.graph.number_of_edges()}个边")
    
    def answer(self, query_text):
        """回答问题
        
        Args:
            query_text: 问题文本
            
        Returns:
            str: 回答
        """
        try:
            # 处理查询文本
            query_text = query_text.encode('utf-8', errors='ignore').decode('utf-8')
            logger.info(f"回答问题: {query_text}")
            
            # 查询知识图谱
            kg_result = None
            if self.use_kg:
                try:
                    kg_result = self.query_kg(query_text)
                    # 同时检索相关文本
                    relevant_texts = self.retrieve_relevant_texts(query_text, top_k=2)
                except Exception as e:
                    logger.error(f"查询知识图谱时出错: {str(e)}", exc_info=True)
            
            # 构建提示
            system_prompt = "你是一个基于马克思主义理论的助手，专注于解释马克思和恩格斯的思想。你的回答应该基于知识图谱和相关文本提供的信息，并明确引用这些信息。"
            
            if kg_result:
                # 添加知识图谱信息
                prompt = f"我将为你提供与问题相关的知识图谱信息和文本片段。\n\n知识图谱信息：\n{kg_result}\n\n"
                
                # 添加相关文本片段（如果有）
                if hasattr(self, 'text_chunks') and self.text_chunks and relevant_texts:
                    prompt += "相关文本片段：\n"
                    for i, text_chunk in enumerate(relevant_texts):
                        prompt += f"[文本 {i+1}] 标题: {text_chunk.title}\n"
                        prompt += f"卷号: {text_chunk.volume}, 文章编号: {text_chunk.article_number}\n"
                        prompt += f"内容: {text_chunk.text[:500]}...\n\n"
                
                prompt += f"请基于上述知识图谱信息和文本片段，详细回答以下问题。如果上述信息不足以回答问题，请明确说明：\n\n问题: {query_text}"
            else:
                prompt = query_text
            
            # 调用LLM API
            try:
                headers = {
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "qwen",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                
                logger.info(f"发送到LLM的提示: {prompt[:200]}...")
                
                response = requests.post(
                    self.llm_api_url,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"调用LLM API时出错: {str(e)}", exc_info=True)
                return f"调用LLM API时出错: {str(e)}"
        except Exception as e:
            logger.error(f"回答问题时出错: {str(e)}", exc_info=True)
            return f"回答问题时出错: {str(e)}"
    
    def query_llm(self, messages):
        """查询LLM
        
        Args:
            messages: 消息列表
            
        Returns:
            str: LLM回复
        """
        try:
            # 确保消息格式符合LLaMA Factory API的要求
            # 只支持用户(user)和助手(assistant)交替的消息
            formatted_messages = []
            
            # 如果第一条消息是系统消息，将其内容添加到第一条用户消息前面
            if messages and messages[0]["role"] == "system":
                system_content = messages[0]["content"]
                # 找到第一条用户消息
                user_idx = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
                if user_idx is not None:
                    messages[user_idx]["content"] = f"{system_content}\n\n{messages[user_idx]['content']}"
            
            # 过滤掉所有系统消息，只保留用户和助手消息
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    formatted_messages.append(msg)
            
            # 确保消息列表以用户消息开始
            if not formatted_messages or formatted_messages[0]["role"] != "user":
                logger.warning("消息列表必须以用户消息开始，添加一个空的用户消息")
                formatted_messages.insert(0, {"role": "user", "content": "你好"})
            
            # 确保用户和助手消息交替出现
            final_messages = [formatted_messages[0]]  # 从第一条用户消息开始
            for i in range(1, len(formatted_messages)):
                if formatted_messages[i]["role"] != final_messages[-1]["role"]:
                    final_messages.append(formatted_messages[i])
                else:
                    # 如果连续两条消息角色相同，合并它们
                    logger.warning(f"发现连续的{formatted_messages[i]['role']}消息，合并内容")
                    final_messages[-1]["content"] += f"\n\n{formatted_messages[i]['content']}"
            
            # 发送请求
            logger.info(f"发送请求到LLM API: {self.llm_api_url}")
            response = requests.post(
                self.llm_api_url,
                json={
                    "model": "Qwen2.5-7B-Instruct",
                    "messages": final_messages,
                    "temperature": 0.7
                },
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 返回LLM回复
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM API返回错误: {str(e)}")
            return f"LLM API错误: {str(e)}"
    
    def answer(self, query_text):
        """回答问题
        
        Args:
            query_text: 用户问题
            
        Returns:
            str: 回答
        """
        try:
            # 处理输入文本，移除可能导致编码问题的字符
            query_text = query_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 查询知识图谱
            kg_result = None
            if self.use_kg:
                kg_result = self.query_kg(query_text)
            
            # 构建提示
            messages = []
            
            # 系统消息
            messages.append({
                "role": "user", 
                "content": "你是一个基于马克思主义理论的助手，专注于回答关于马克思、恩格斯及其著作的问题。"
            })
            
            # 添加助手回复
            messages.append({
                "role": "assistant",
                "content": "我是一个基于马克思主义理论的助手，专注于解释马克思和恩格斯的思想。有什么可以帮助您的吗？"
            })
            
            # 如果有知识图谱结果，添加到提示中
            if kg_result:
                prompt = f"""我将为你提供一些相关的知识图谱信息，这些信息来自马克思恩格斯的著作。
请基于这些信息回答用户的问题，同时结合你自己的知识。
如果知识图谱信息与问题相关，请优先使用这些信息。

知识图谱信息:
{kg_result}

请回答以下问题，并明确指出你的回答中哪些部分是基于知识图谱的，哪些是基于你自己的知识的。"""
                
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                messages.append({
                    "role": "assistant",
                    "content": "我理解了。我会基于提供的知识图谱信息和我自己的知识来回答问题。"
                })
            
            # 添加用户问题
            messages.append({"role": "user", "content": query_text})
            
            # 查询LLM - 每次查询都使用新的消息列表，不保留历史
            return self.query_llm(messages)
        except Exception as e:
            logger.error(f"回答问题时出错: {str(e)}", exc_info=True)
            return f"回答问题时出错: {str(e)}"
    
    # ... (其余代码保持不变)

def query_dmx_api(query: str, kg_enhancer: Optional[KGEnhancedLLM] = None) -> str:
    """直接使用DMX API查询，不依赖本地API服务器"""
    # 获取知识图谱上下文
    kg_context = ""
    if kg_enhancer:
        kg_result = kg_enhancer.query_kg(query)
        if kg_result:
            kg_context = "以下是与问题相关的知识图谱信息：\n" + kg_result + "\n\n请基于上述信息回答问题。"
    
    # 构建提示
    if kg_context:
        prompt = f"{kg_context}\n\n问题: {query}"
    else:
        prompt = query
    
    # 调用DMX API
    api_key = os.environ.get("DMX_API_KEY")
    if not api_key:
        return "错误：未设置DMX_API_KEY环境变量"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一个基于马克思主义理论的助手，专注于解释马克思和恩格斯的思想。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://www.dmxapi.cn/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API调用错误: {str(e)}"

def run_interactive_mode(llm_api_url, use_kg, no_api):
    """运行交互式模式"""
    kg_llm = KGEnhancedLLM(llm_api_url=llm_api_url, use_kg=use_kg)
    
    print("\n=== 马克思恩格斯知识图谱增强LLM ===")
    print(f"知识图谱增强: {'启用' if use_kg else '禁用'}")
    print(f"使用DMX API: {'是' if no_api else '否'}")
    print("输入您的问题，或输入'exit'退出\n")
    
    while True:
        try:
            user_query = input("\n问题: ")
            if user_query.lower() in ['exit', 'quit', '退出']:
                break
                
            if not user_query.strip():
                continue
                
            logger.info(f"用户查询: {user_query}")
            
            # 获取知识图谱和文本信息
            kg_info = None
            text_info = None
            if use_kg:
                try:
                    # 查询知识图谱
                    kg_info = kg_llm.query_kg(user_query)
                    # 检索相关文本
                    text_chunks = kg_llm.retrieve_relevant_texts(user_query, top_k=2)
                    
                    # 显示提供给LLM的额外信息
                    print("\n=== 提供给LLM的知识图谱信息 ===")
                    print(kg_info if kg_info else "无相关知识图谱信息")
                    
                    if text_chunks:
                        print("\n=== 提供给LLM的相关文本片段 ===")
                        for i, chunk in enumerate(text_chunks):
                            print(f"\n[文本 {i+1}] 标题: {chunk.title}")
                            print(f"卷号: {chunk.volume}, 文章编号: {chunk.article_number}")
                            print(f"内容: {chunk.text[:300]}...")
                    else:
                        print("\n无相关文本片段")
                except Exception as e:
                    logger.error(f"获取知识图谱和文本信息时出错: {str(e)}", exc_info=True)
                    print(f"\n获取知识图谱和文本信息时出错: {str(e)}")
            
            # 执行查询
            if no_api:
                response = query_dmx_api(user_query, kg_llm)
            else:
                response = kg_llm.answer(user_query)
            
            print(f"\n=== LLM回答 ===")
            print(response)
            
        except KeyboardInterrupt:
            print("\n查询已中断")
            break
        except Exception as e:
            logger.error(f"查询时出错: {str(e)}", exc_info=True)
            print(f"\n查询时出错: {str(e)}")
    
    print("\n感谢使用马克思恩格斯知识图谱增强LLM！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="马克思恩格斯知识图谱增强LLM")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="LLM API的URL")
    parser.add_argument("--no-kg", action="store_true", help="禁用知识图谱增强")
    parser.add_argument("--no-api", action="store_true", help="不使用API，直接使用DMX API")
    parser.add_argument("--query", type=str, help="单次查询的问题")
    
    args = parser.parse_args()
    
    if args.query:
        # 单次查询模式
        kg_llm = KGEnhancedLLM(llm_api_url=args.api_url, use_kg=not args.no_kg)
        if args.no_api:
            response = query_dmx_api(args.query, kg_llm)
            print(f"\n回答: {response}")
        else:
            response = kg_llm.answer(args.query)
            print(f"\n回答: {response}")
    else:
        # 交互式模式
        run_interactive_mode(args.api_url, not args.no_kg, args.no_api)

if __name__ == "__main__":
    main()

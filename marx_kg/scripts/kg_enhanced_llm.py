"""
马克思恩格斯知识图谱增强LLM脚本
将微调的Qwen模型与GraphRAG知识图谱结合
"""
import os
import sys
import json
import logging
import requests
import asyncio
import tempfile
import pickle
import pandas as pd
import networkx as nx
import tiktoken
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Any, Tuple

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_enhanced_llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        
        if self.use_kg:
            # 初始化知识图谱查询引擎
            self.initialize_kg_query()
        
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
                    description=row.get('description', ''),
                    description_embedding=None,
                    name_embedding=None
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
            
            # 创建LanceDB向量存储
            vector_store = LanceDBVectorStore(collection_name="descriptions")
            db_uri = os.path.join(temp_dir)
            vector_store.connect(db_uri=db_uri)
            
            # 创建文档集合
            documents = []
            for entity in entities:
                # 创建一个默认的向量（全0向量）
                default_vector = [0.0] * 1536  # 使用1536维向量，这是常见的嵌入维度
                
                doc = VectorStoreDocument(
                    id=entity.id,
                    text=entity.description,
                    vector=default_vector,
                    attributes={
                        "title": entity.title,
                        "type": entity.type
                    }
                )
                documents.append(doc)
            
            # 加载文档到向量存储
            logger.info(f"加载{len(documents)}个实体描述到向量存储")
            vector_store.load_documents(documents)
            
            # 创建嵌入模型配置
            embedding_model_config = LanguageModelConfig(
                type="openai_embedding",
                model="text-embedding-ada-002",
                encoding_model="cl100k_base",
                api_key=os.environ.get("DMX_API_KEY"),
                api_base="https://www.dmxapi.cn/v1"
            )

            # 创建聊天模型配置
            chat_model_config = LanguageModelConfig(
                type="openai_chat",  # 使用正确的类型
                model="gpt-3.5-turbo",
                encoding_model="cl100k_base",
                api_key=os.environ.get("DMX_API_KEY"),
                api_base="https://www.dmxapi.cn/v1"
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
                    "default_embedding_model": embedding_model_config,
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
                callbacks=QueryCallbacks()
            )
            
            logger.info("GraphRAG查询引擎初始化完成")
            
        except Exception as e:
            logger.error(f"初始化知识图谱查询引擎时出错: {str(e)}", exc_info=True)
            self.search_engine = None
            self.graph = None
    
    def query_kg(self, query_text):
        """查询知识图谱
        
        Args:
            query_text: 查询文本
            
        Returns:
            str: 知识图谱查询结果
        """
        if not self.use_kg:
            return None
            
        try:
            logger.info(f"查询知识图谱: {query_text}")
            
            # 尝试使用GraphRAG查询引擎
            if hasattr(self, 'search_engine') and self.search_engine is not None:
                try:
                    # 使用search方法查询知识图谱
                    result = asyncio.run(self.search_engine.search(query_text))
                    if result and hasattr(result, 'context_text'):
                        logger.info(f"GraphRAG查询结果: {result.context_text}")
                        return result.context_text
                    else:
                        logger.warning("GraphRAG查询返回的结果为空或格式不正确")
                        # 回退到NetworkX查询
                        raise Exception("回退到NetworkX查询")
                except Exception as e:
                    logger.error(f"使用GraphRAG查询时出错: {str(e)}", exc_info=True)
                    # 如果GraphRAG查询失败，回退到NetworkX查询
                    raise Exception("回退到NetworkX查询")
            
            # 回退到NetworkX查询
            if hasattr(self, 'graph') and self.graph is not None:
                # 使用NetworkX直接查询
                # 这是一个简化的实现，只查找与查询文本相关的实体和关系
                query_keywords = query_text.lower().split()
                
                # 查找相关节点
                related_nodes = []
                for node, data in self.graph.nodes(data=True):
                    node_text = f"{node} {data.get('description', '')}".lower()
                    if any(keyword in node_text for keyword in query_keywords):
                        related_nodes.append(node)
                
                if not related_nodes:
                    return "知识图谱查询结果:\n\n未找到与查询相关的实体。\n\n未找到与查询相关的关系。"
                
                # 查找相关节点之间的关系
                related_edges = []
                for source in related_nodes:
                    for target in related_nodes:
                        if source != target and self.graph.has_edge(source, target):
                            edge_data = self.graph.get_edge_data(source, target)
                            related_edges.append((source, target, edge_data))
                
                # 格式化结果
                result = "知识图谱查询结果:\n\n相关实体:\n"
                for node in related_nodes:
                    node_data = self.graph.nodes[node]
                    result += f"- {node}: {node_data.get('description', '无描述')}\n"
                
                result += "\n相关关系:\n"
                if related_edges:
                    for source, target, edge_data in related_edges:
                        result += f"- {source} -> {target}: {edge_data.get('description', '无描述')}\n"
                else:
                    result += "未找到相关实体之间的关系。\n"
                
                logger.info(f"NetworkX查询结果: {result}")
                return result
            
            return "知识图谱查询失败，无法获取结果。"
        except Exception as e:
            logger.error(f"查询知识图谱时出错: {str(e)}", exc_info=True)
            return f"查询知识图谱时出错: {str(e)}"
    
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
            
            # 执行查询
            if no_api:
                response = query_dmx_api(user_query, kg_llm)
            else:
                response = kg_llm.answer(user_query)
            
            print(f"\n回答: {response}")
            
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

"""
马克思恩格斯知识图谱增强LLM脚本
将微调的Qwen模型与GraphRAG知识图谱结合
"""
import os
import sys
import json
import logging
import requests
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
from graphrag.data_model.text_unit import TextUnit
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.language_model.manager import ModelManager
from graphrag.query.factory import get_local_search_engine
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.vector_stores.lancedb import LanceDBVectorStore

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
            # 加载配置
            config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
            logger.info(f"加载配置文件: {config_path}")
            config = load_config(PROJECT_ROOT, config_path)
            
            # 加载知识图谱数据
            kg_dir = PROJECT_ROOT / "data" / "kg"
            entities_path = kg_dir / "entities.csv"
            relationships_path = kg_dir / "relationships.csv"
            
            logger.info(f"加载实体数据: {entities_path}")
            entities_df = pd.read_csv(entities_path)
            
            logger.info(f"加载关系数据: {relationships_path}")
            relationships_df = pd.read_csv(relationships_path)
            
            # 转换为GraphRAG数据模型
            entities = []
            for _, row in entities_df.iterrows():
                entity = Entity(
                    id=row['title'],  # 使用title作为id
                    title=row['title'],
                    type=row['type'],
                    description=row['description'],
                    document_id=''  # 没有document_id，使用空字符串
                )
                entities.append(entity)
            
            relationships = []
            for _, row in relationships_df.iterrows():
                relationship = Relationship(
                    id=f"{row['source']}-{row['target']}",  # 使用source-target作为id
                    source=row['source'],
                    target=row['target'],
                    description=row['description'],
                    document_id=''  # 没有document_id，使用空字符串
                )
                relationships.append(relationship)
            
            # 创建简单的文本单元
            text_units = []
            
            # 创建LanceDB向量存储
            # 创建临时目录用于LanceDB数据库
            import tempfile
            db_dir = tempfile.mkdtemp()
            logger.info(f"创建LanceDB数据库目录: {db_dir}")
            
            # 初始化LanceDB向量存储
            description_embedding_store = LanceDBVectorStore(
                collection_name="descriptions",
                db_uri=f"uri://{db_dir}"
            )
            
            # 连接到向量存储
            description_embedding_store.connect(db_uri=f"uri://{db_dir}")
            
            # 由于我们没有实际的社区报告，创建一个空列表
            reports = []
            
            # 由于我们没有实际的协变量，创建一个空字典
            covariates = {}
            
            # 初始化本地搜索引擎
            self.search_engine = get_local_search_engine(
                config=config,
                reports=reports,
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                covariates=covariates,
                response_type="multiple paragraphs",
                description_embedding_store=description_embedding_store,
                callbacks=[QueryCallbacks()]
            )
            
            logger.info("GraphRAG查询引擎初始化完成")
        except Exception as e:
            logger.error(f"初始化GraphRAG查询引擎时出错: {str(e)}", exc_info=True)
            self.search_engine = None
            
            # 回退到直接加载NetworkX图谱
            try:
                kg_path = PROJECT_ROOT / "data" / "kg" / "knowledge_graph.pickle"
                logger.info(f"回退到直接加载NetworkX图谱: {kg_path}")
                
                with open(kg_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"NetworkX图谱加载完成，包含 {len(self.graph.nodes)} 个节点和 {len(self.graph.edges)} 条边")
            except Exception as e2:
                logger.error(f"加载NetworkX图谱时出错: {str(e2)}", exc_info=True)
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
                    # 使用GraphRAG的本地搜索引擎
                    result = self.search_engine.search(query_text)
                    logger.info(f"GraphRAG查询结果: {result.context_text}")
                    return result.context_text
                except Exception as e:
                    logger.error(f"使用GraphRAG查询时出错: {str(e)}", exc_info=True)
                    # 如果GraphRAG查询失败，回退到NetworkX查询
            
            # 回退到NetworkX查询
            if hasattr(self, 'graph') and self.graph is not None:
                # 使用简单的关键词匹配查询知识图谱
                keywords = query_text.lower().split()
                relevant_nodes = []
                
                # 查找与关键词匹配的节点
                for node in self.graph.nodes:
                    node_text = str(node).lower()
                    if any(keyword in node_text for keyword in keywords):
                        node_data = self.graph.nodes[node]
                        relevant_nodes.append({
                            "entity": node,
                            "type": node_data.get("type", "未知"),
                            "description": node_data.get("description", "无描述")
                        })
                
                # 如果找到了相关节点，查找它们之间的关系
                relevant_relationships = []
                if relevant_nodes:
                    for node_info in relevant_nodes:
                        node = node_info["entity"]
                        # 查找以该节点为起点的边
                        for target in self.graph.successors(node):
                            edge_data = self.graph.edges[node, target]
                            relevant_relationships.append({
                                "source": node,
                                "target": target,
                                "description": edge_data.get("description", "无描述"),
                                "strength": edge_data.get("strength", 0.5)
                            })
                        
                        # 查找以该节点为终点的边
                        for source in self.graph.predecessors(node):
                            edge_data = self.graph.edges[source, node]
                            relevant_relationships.append({
                                "source": source,
                                "target": node,
                                "description": edge_data.get("description", "无描述"),
                                "strength": edge_data.get("strength", 0.5)
                            })
                
                # 格式化结果
                result = "知识图谱查询结果:\n\n"
                
                if relevant_nodes:
                    result += "相关实体:\n"
                    for node_info in relevant_nodes:
                        result += f"- {node_info['entity']} (类型: {node_info['type']}): {node_info['description']}\n"
                    result += "\n"
                else:
                    result += "未找到与查询相关的实体。\n\n"
                
                if relevant_relationships:
                    result += "相关关系:\n"
                    for rel in relevant_relationships:
                        result += f"- {rel['source']} -> {rel['target']}: {rel['description']} (强度: {rel['strength']})\n"
                    result += "\n"
                else:
                    result += "未找到与查询相关的关系。\n"
                
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
        # 构建系统提示
        system_prompt = "你是一个基于马克思主义理论的助手，专注于回答关于马克思、恩格斯及其著作的问题。"
        
        # 如果启用了知识图谱，查询知识图谱
        kg_result = None
        if self.use_kg:
            kg_result = self.query_kg(query_text)
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 如果有知识图谱结果，添加到系统提示中
        if kg_result:
            kg_prompt = f"""我将为你提供一些相关的知识图谱信息，这些信息来自马克思恩格斯的著作。
请基于这些信息回答用户的问题，同时结合你自己的知识。
如果知识图谱信息与问题相关，请优先使用这些信息。

知识图谱信息:
{kg_result}

请回答以下问题，并明确指出你的回答中哪些部分是基于知识图谱的，哪些是基于你自己的知识的。"""
            
            messages.append({"role": "system", "content": kg_prompt})
        
        # 添加用户问题
        messages.append({"role": "user", "content": query_text})
        
        # 查询LLM
        return self.query_llm(messages)

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

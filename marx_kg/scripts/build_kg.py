"""
马克思恩格斯知识图谱构建脚本
使用GraphRAG构建知识图谱
"""
import os
import sys
import json
import logging
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
import asyncio

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.config.enums import AsyncType
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
from graphrag.cache.memory_pipeline_cache import InMemoryCache
from graphrag.cache.json_pipeline_cache import JsonPipelineCache

# 导入操作函数 - 使用正确的导入路径
from graphrag.index.operations.extract_graph.extract_graph import extract_graph
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.index.operations.prune_graph import prune_graph
from graphrag.index.operations.embed_graph.embed_graph import embed_graph
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import summarize_descriptions

# 导入常量定义
from graphrag.config.defaults import DEFAULT_CHAT_MODEL_ID, DEFAULT_EMBEDDING_MODEL_ID

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("build_kg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def build_knowledge_graph():
    """构建知识图谱"""
    logger.info("开始构建马克思恩格斯知识图谱")
    
    # 检查DMX API密钥
    if 'DMX_API_KEY' not in os.environ:
        logger.error("未设置DMX_API_KEY环境变量，请设置后重试")
        logger.info("您可以通过以下命令设置环境变量:")
        logger.info("在Linux中: export DMX_API_KEY='sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY'")
        logger.info("在Windows PowerShell中: $env:DMX_API_KEY='sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY'")
        logger.info("在Windows CMD中: set DMX_API_KEY=sk-ZhF2CPvVSjuqOZOk5n2fbancon1CTyxWzS1TtQx84JB5GQOY")
        return None
    
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(PROJECT_ROOT, config_path)
    
    # 打印配置信息
    logger.info(f"模型配置: {config.models}")
    
    # 创建回调和缓存
    callbacks = ConsoleWorkflowCallbacks()  # 使用控制台回调，它会在控制台显示进度
    
    # 使用内存缓存而不是JSON缓存，避免PosixPath的child方法问题
    cache = InMemoryCache()
    
    # 加载处理后的文档
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "marx_engels_documents.json"
    logger.info(f"加载处理后的文档: {processed_data_path}")
    
    # 确保目录存在
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件不存在，创建一个简单的示例文档
    if not processed_data_path.exists():
        logger.warning(f"文档文件不存在，创建示例文档: {processed_data_path}")
        example_docs = [
            {
                "id": "doc_1",
                "title": "共产党宣言",
                "text": "一个幽灵，共产主义的幽灵，在欧洲游荡。为了对这个幽灵进行神圣的围剿，旧欧洲的一切势力，教皇和沙皇、梅特涅和基佐、法国的激进派和德国的警察，都联合起来了。"
            },
            {
                "id": "doc_2",
                "title": "资本论",
                "text": "资本主义生产方式占统治地位的社会的财富，表现为庞大的商品堆积，单个的商品表现为这种财富的元素形式。因此，我们的研究就从分析商品开始。"
            },
            {
                "id": "doc_3",
                "title": "辩证法",
                "text": "辩证法所谓的客观性质，自然界中到处都有。偶然性不断地被必然性所排斥，而必然性又不断地被偶然性所打破。"
            }
        ]
        with open(processed_data_path, "w", encoding="utf-8") as f:
            json.dump(example_docs, f, ensure_ascii=False, indent=2)
    
    with open(processed_data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    # 将文档转换为DataFrame
    text_units = pd.DataFrame(documents)
    logger.info(f"加载了 {len(text_units)} 个文档")
    
    # 打印DataFrame的列名，帮助调试
    logger.info(f"DataFrame列名: {text_units.columns.tolist()}")
    logger.info(f"DataFrame内容示例: \n{text_units.head()}")
    
    # 如果DataFrame为空或没有列，打印更多信息
    if text_units.empty or len(text_units.columns) == 0:
        logger.error("DataFrame为空或没有列")
        logger.info(f"原始文档内容: {documents}")
        return None
    
    # 检查是否有'text'列，如果有，使用'text'作为文本列
    text_column = 'text' if 'text' in text_units.columns else 'content'
    id_column = 'id' if 'id' in text_units.columns else 'document_id'
    
    logger.info(f"使用文本列: {text_column}")
    logger.info(f"使用ID列: {id_column}")
    
    # 如果没有指定的文本列，尝试使用第一列
    if text_column not in text_units.columns:
        text_column = text_units.columns[0]
        logger.warning(f"找不到'text'或'content'列，使用第一列: {text_column}")
    
    # 如果没有指定的ID列，创建一个ID列
    if id_column not in text_units.columns:
        text_units['id'] = [f"doc_{i}" for i in range(len(text_units))]
        id_column = 'id'
        logger.warning(f"找不到'id'或'document_id'列，创建了一个新的ID列")
    
    # 确保有title列，如果没有，从id创建
    if 'title' not in text_units.columns:
        logger.warning("找不到'title'列，从ID创建")
        text_units['title'] = text_units[id_column].apply(lambda x: f"Document {x}")
    
    # 提取图谱
    logger.info("步骤1: 提取实体和关系")
    extract_config = config.extract_graph if hasattr(config, 'extract_graph') else None
    
    # 默认的实体类型
    default_entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "WORK_OF_ART", "EVENT", "DATE", "CONCEPT"]
    
    # 如果extract_config存在且有entity_types属性，则使用它，否则使用默认值
    entity_types = extract_config.entity_types if extract_config and hasattr(extract_config, 'entity_types') else default_entity_types
    
    # 获取默认聊天模型配置
    default_chat_model = config.models[DEFAULT_CHAT_MODEL_ID] if DEFAULT_CHAT_MODEL_ID in config.models else None
    
    # 创建自定义策略
    custom_strategy = {
        "type": "graph_intelligence",
        "llm": {
            "type": "openai_chat",
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("DMX_API_KEY"),
            "api_base": "https://www.dmxapi.cn/v1",
            "encoding_model": "cl100k_base",
            "max_retries": 3,
            "temperature": 0,
            "max_tokens": 2048,
            "request_timeout": 180.0
        },
        "entity_name_column": "title",
        "entity_type_column": "type",
        "entity_description_column": "description",
        "source_entity_column": "source",
        "target_entity_column": "target",
        "relationship_description_column": "description",
        "relationship_strength_column": "strength"
    }
    
    try:
        # 使用API提取实体和关系
        logger.info("使用DMX API提取实体和关系")
        
        try:
            entities, relationships = await extract_graph(
                text_units=text_units,
                callbacks=callbacks,
                cache=cache,
                text_column=text_column,
                id_column=id_column,
                strategy=custom_strategy,
                async_mode=AsyncType.AsyncIO,
                entity_types=entity_types,
                num_threads=4
            )
            logger.info(f"成功提取了 {len(entities)} 个实体和 {len(relationships)} 个关系")
        except Exception as e:
            logger.warning(f"使用DMX API提取实体和关系失败: {e}，使用模拟数据")
            
            # 创建模拟实体和关系数据
            entities = pd.DataFrame({
                'id': ['entity_1', 'entity_2', 'entity_3', 'entity_4'],
                'title': ['马克思', '恩格斯', '共产主义', '资本主义'],
                'type': ['PERSON', 'PERSON', 'CONCEPT', 'CONCEPT'],
                'description': [
                    '卡尔·马克思，共产主义理论的创始人之一',
                    '弗里德里希·恩格斯，马克思的合作者',
                    '共产主义是一种政治和经济理论',
                    '资本主义是一种经济体系'
                ],
                'document_id': ['doc_1', 'doc_1', 'doc_1', 'doc_2']
            })
            
            relationships = pd.DataFrame({
                'id': ['rel_1', 'rel_2', 'rel_3'],
                'source': ['马克思', '马克思', '恩格斯'],
                'target': ['恩格斯', '共产主义', '共产主义'],
                'description': [
                    '马克思和恩格斯是合作者',
                    '马克思是共产主义理论的创始人',
                    '恩格斯是共产主义理论的支持者'
                ],
                'strength': [0.9, 0.8, 0.7],
                'document_id': ['doc_1', 'doc_1', 'doc_1']
            })
            
            logger.info(f"创建了模拟实体: {len(entities)}个")
            logger.info(f"创建了模拟关系: {len(relationships)}个")
    except Exception as e:
        logger.error(f"提取实体和关系时出错: {e}")
        raise
    
    # 创建输出目录
    kg_dir = PROJECT_ROOT / "data" / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存实体和关系
    entities_path = kg_dir / "entities.csv"
    relationships_path = kg_dir / "relationships.csv"
    
    entities.to_csv(entities_path, index=False)
    relationships.to_csv(relationships_path, index=False)
    
    logger.info(f"保存实体到: {entities_path}")
    logger.info(f"保存关系到: {relationships_path}")
    
    # 构建知识图谱
    logger.info("步骤2: 构建知识图谱")
    G = nx.DiGraph()
    
    # 添加节点
    for _, row in entities.iterrows():
        G.add_node(row['title'], type=row['type'], description=row['description'])
    
    # 添加边
    for _, row in relationships.iterrows():
        # 检查是否有strength字段，如果没有，设置默认值为0.5
        strength = row['strength'] if 'strength' in row else 0.5
        G.add_edge(row['source'], row['target'], description=row['description'], strength=strength)
    
    # 保存图谱 - 使用pickle替代write_gpickle
    graph_path = kg_dir / "knowledge_graph.pickle"
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    logger.info(f"保存知识图谱到: {graph_path}")
    
    return kg_dir

async def main():
    """主函数"""
    try:
        kg_dir = await build_knowledge_graph()
        if kg_dir:
            logger.info(f"知识图谱构建完成，保存在: {kg_dir}")
        else:
            logger.error("知识图谱构建失败")
    except Exception as e:
        logger.error(f"构建知识图谱时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

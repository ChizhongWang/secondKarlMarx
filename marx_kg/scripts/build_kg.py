"""
马克思恩格斯知识图谱构建脚本
使用GraphRAG构建知识图谱
"""
import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.config.enums import AsyncType
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.cache.pipeline_cache import PipelineCache

# 导入操作函数而不是类
from graphrag.index.operations.extract_graph.extract_graph import extract_graph
from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.index.operations.prune_graph.prune_graph import prune_graph
from graphrag.index.operations.embed_graph.embed_graph import embed_graph
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import summarize_descriptions

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
    
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(PROJECT_ROOT, config_path)
    
    # 创建回调和缓存
    callbacks = WorkflowCallbacks()
    cache = PipelineCache(config.cache)
    
    # 加载处理后的文档
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "marx_engels_documents.json"
    logger.info(f"加载处理后的文档: {processed_data_path}")
    
    with open(processed_data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    # 将文档转换为DataFrame
    text_units = pd.DataFrame(documents)
    logger.info(f"加载了 {len(text_units)} 个文档")
    
    # 提取图谱
    logger.info("步骤1: 提取实体和关系")
    extract_config = config.extract_graph if hasattr(config, 'extract_graph') else {}
    entity_types = extract_config.get('entity_types', ["PERSON", "ORGANIZATION", "LOCATION", "WORK_OF_ART", "EVENT", "DATE", "CONCEPT"])
    
    entities, relationships = await extract_graph(
        text_units=text_units,
        callbacks=callbacks,
        cache=cache,
        text_column="content",
        id_column="id",
        strategy=extract_config,
        async_mode=AsyncType.AsyncIO,
        entity_types=entity_types,
        num_threads=4
    )
    
    logger.info(f"提取了 {len(entities)} 个实体和 {len(relationships)} 个关系")
    
    # 修剪图谱
    logger.info("步骤2: 修剪图谱")
    prune_config = config.prune_graph if hasattr(config, 'prune_graph') else {}
    
    pruned_entities, pruned_relationships = await prune_graph(
        entities=entities,
        relationships=relationships,
        callbacks=callbacks,
        cache=cache,
        config=prune_config
    )
    
    logger.info(f"修剪后保留 {len(pruned_entities)} 个实体和 {len(pruned_relationships)} 个关系")
    
    # 生成嵌入
    logger.info("步骤3: 生成文本嵌入")
    embed_text_config = config.embed_text if hasattr(config, 'embed_text') else {}
    
    await embed_text(
        text_units=text_units,
        callbacks=callbacks,
        cache=cache,
        text_column="content",
        id_column="id",
        config=embed_text_config
    )
    
    # 嵌入图谱
    logger.info("步骤4: 嵌入图谱")
    embed_graph_config = config.embed_graph if hasattr(config, 'embed_graph') else {}
    
    await embed_graph(
        entities=pruned_entities,
        relationships=pruned_relationships,
        callbacks=callbacks,
        cache=cache,
        config=embed_graph_config
    )
    
    # 生成摘要描述
    logger.info("步骤5: 生成实体摘要描述")
    summarize_config = config.summarize_descriptions if hasattr(config, 'summarize_descriptions') else {}
    
    await summarize_descriptions(
        entities=pruned_entities,
        relationships=pruned_relationships,
        callbacks=callbacks,
        cache=cache,
        config=summarize_config
    )
    
    logger.info("知识图谱构建完成")
    
    # 返回图谱路径
    kg_output_dir = PROJECT_ROOT / "kg" / "output"
    return kg_output_dir

async def main():
    """主函数"""
    try:
        kg_dir = await build_knowledge_graph()
        logger.info(f"知识图谱已保存到: {kg_dir}")
        print(f"\n知识图谱构建成功！\n图谱文件保存在: {kg_dir}")
    except Exception as e:
        logger.error(f"构建知识图谱时出错: {str(e)}", exc_info=True)
        print(f"\n构建知识图谱时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

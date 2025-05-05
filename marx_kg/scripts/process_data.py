"""
马克思恩格斯全集数据处理脚本
将原始文本转换为GraphRAG可处理的格式
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def ensure_dirs():
    """确保目录存在"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"确保目录存在: {PROCESSED_DATA_DIR}")

def list_raw_files() -> List[Path]:
    """列出原始数据文件"""
    files = list(RAW_DATA_DIR.glob("*.txt"))
    logger.info(f"找到 {len(files)} 个原始文本文件")
    return files

def process_text_file(file_path: Path) -> List[Dict[str, Any]]:
    """处理单个文本文件
    
    Args:
        file_path: 文本文件路径
        
    Returns:
        处理后的文档列表，每个文档是一个字典
    """
    logger.info(f"处理文件: {file_path.name}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取元数据（假设文件名格式为：作者_作品名_年份.txt）
    filename = file_path.stem
    parts = filename.split('_')
    
    metadata = {
        "filename": file_path.name,
        "author": parts[0] if len(parts) > 0 else "未知",
        "title": parts[1] if len(parts) > 1 else file_path.stem,
        "year": parts[2] if len(parts) > 2 else "未知",
        "source": "马克思恩格斯全集"
    }
    
    # 分割文本为章节（假设章节以数字标题开始）
    chapters = re.split(r'\n(?=第[一二三四五六七八九十百千万]+章|\d+\.)', content)
    
    documents = []
    for i, chapter in enumerate(chapters):
        if not chapter.strip():
            continue
            
        # 创建文档
        doc_id = f"{file_path.stem}_chapter_{i+1}"
        document = {
            "id": doc_id,
            "text": chapter.strip(),
            "metadata": {
                **metadata,
                "chapter": i+1,
                "chapter_title": chapter.split('\n')[0] if chapter else ""
            }
        }
        documents.append(document)
    
    logger.info(f"从文件 {file_path.name} 中提取了 {len(documents)} 个章节")
    return documents

def process_all_files():
    """处理所有原始文件"""
    ensure_dirs()
    files = list_raw_files()
    
    all_documents = []
    for file_path in files:
        documents = process_text_file(file_path)
        all_documents.extend(documents)
    
    # 保存为JSON文件
    output_file = PROCESSED_DATA_DIR / "marx_engels_documents.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成，共生成 {len(all_documents)} 个文档，保存到 {output_file}")

def main():
    """主函数"""
    logger.info("开始处理马克思恩格斯全集数据")
    process_all_files()
    logger.info("数据处理完成")

if __name__ == "__main__":
    main()

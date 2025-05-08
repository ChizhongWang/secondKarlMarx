"""
马克思主义辞典数据处理脚本
将HTML格式的马克思主义辞典转换为GraphRAG可处理的格式，并与现有的马恩全集JSON文件合并
"""
import os
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_dictionary.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
COLLECTION_WORKS_DIR = Path("d:/王驰中的经历/研究生/研一下课程/自然语言处理/secondKarlMarx/marx_collection_works")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def ensure_dirs():
    """确保目录存在"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"确保目录存在: {PROCESSED_DATA_DIR}")

def extract_dictionary_entry(html_content, entry_id):
    """从HTML内容中提取词条"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 查找指定ID的锚点
    anchor = soup.find('a', {'name': entry_id})
    if not anchor:
        return None
    
    # 找到词条标题
    title_element = anchor.find_next('p', {'class': 'title2'})
    if not title_element:
        return None
    
    title = title_element.get_text().strip()
    
    # 提取词条内容
    content = ""
    current = title_element.find_next('p')
    
    # 收集内容直到下一个词条或结束
    while current and not (current.name == 'p' and 'title2' in current.get('class', [])):
        if current.name == 'p':
            content += current.get_text().strip() + "\n\n"
        current = current.find_next()
    
    return {
        "title": title,
        "content": content.strip()
    }

def process_dictionary_html():
    """处理马克思主义辞典HTML文件"""
    logger.info("开始处理马克思主义辞典HTML文件")
    
    dictionary_dir = COLLECTION_WORKS_DIR / "dictionary"
    html_files = list(dictionary_dir.glob("*.html"))
    logger.info(f"找到 {len(html_files)} 个HTML文件")
    
    all_entries = []
    
    for html_file in html_files:
        try:
            logger.info(f"处理文件: {html_file.name}")
            
            with open(html_file, 'r', encoding='gb2312', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 找到所有词条链接
            entry_links = soup.find_all('a', style='text-decoration: none', href=lambda href: href and href.startswith('#'))
            
            logger.info(f"在文件 {html_file.name} 中找到 {len(entry_links)} 个词条链接")
            
            # 提取每个词条
            for link in entry_links:
                entry_id = link['href'][1:]  # 去掉#号
                entry_text = link.get_text().strip()
                
                # 提取词条内容
                entry_data = extract_dictionary_entry(html_content, entry_id)
                
                if entry_data:
                    # 创建词条文档
                    document = {
                        "id": f"dictionary_{html_file.stem}_{entry_id}",
                        "text": entry_data["content"],
                        "metadata": {
                            "title": entry_data["title"],
                            "file": html_file.name,
                            "entry_id": entry_id,
                            "source": "马克思主义辞典"
                        }
                    }
                    
                    all_entries.append(document)
            
        except Exception as e:
            logger.error(f"处理文件 {html_file} 时出错: {e}")
    
    logger.info(f"从马克思主义辞典中提取了 {len(all_entries)} 个词条")
    return all_entries

def main():
    """主函数"""
    logger.info("开始处理马克思主义辞典数据并与马恩全集合并")
    
    ensure_dirs()
    
    # 处理马克思主义辞典
    dictionary_docs = process_dictionary_html()
    logger.info(f"处理了 {len(dictionary_docs)} 个马克思主义辞典词条")
    
    # 保存辞典数据
    dictionary_file = PROCESSED_DATA_DIR / "dictionary_documents.json"
    with open(dictionary_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"保存辞典数据到: {dictionary_file}")
    
    # 准备合并后的输出文件
    output_file = PROCESSED_DATA_DIR / "marx_engels_documents.json"
    
    # 检查是否已存在处理后的马恩全集文件
    existing_file = PROCESSED_DATA_DIR / "marx_engels_documents.json"
    if existing_file.exists():
        # 如果存在，读取并添加辞典数据
        logger.info(f"发现现有的处理文件: {existing_file}")
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_docs = json.load(f)
        
        # 合并文档
        all_documents = existing_docs + dictionary_docs
        logger.info(f"合并后共有 {len(all_documents)} 个文档")
        
        # 保存合并后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)
        logger.info(f"保存合并后的文档到: {output_file}")
    else:
        # 如果不存在，创建一个新的处理文件
        # 首先检查是否有原始的马恩全集JSON文件
        collection_works_files = list((COLLECTION_WORKS_DIR / "collection_works_chunks").glob("**/*.json"))
        
        if collection_works_files:
            logger.info(f"找到 {len(collection_works_files)} 个马恩全集JSON文件")
            
            # 处理马恩全集JSON文件
            collection_works_docs = []
            for json_file in collection_works_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 提取元数据和文本
                    metadata = data.get("metadata", {})
                    raw_text = data.get("raw_text", "")
                    
                    # 创建文档
                    document = {
                        "id": f"{json_file.parent.name}_{json_file.stem}",
                        "text": raw_text,
                        "metadata": {
                            **metadata,
                            "source": "马克思恩格斯全集"
                        }
                    }
                    
                    collection_works_docs.append(document)
                    
                except Exception as e:
                    logger.error(f"处理文件 {json_file} 时出错: {e}")
            
            logger.info(f"处理了 {len(collection_works_docs)} 个马恩全集文档")
            
            # 合并文档
            all_documents = collection_works_docs + dictionary_docs
            
            # 保存合并后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            logger.info(f"保存合并后的文档到: {output_file}")
        else:
            # 如果没有找到马恩全集文件，只保存辞典数据
            logger.warning("未找到马恩全集JSON文件，只保存辞典数据")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dictionary_docs, f, ensure_ascii=False, indent=2)
            logger.info(f"保存辞典文档到: {output_file}")
    
    logger.info("处理完成")

if __name__ == "__main__":
    main()

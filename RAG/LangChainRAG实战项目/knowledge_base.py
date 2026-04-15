"""
知识库更新服务
"""
import os
import hashlib
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        processed_md5_list = f.read().splitlines()
    
    return md5_str in processed_md5_list


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(text: str,encoding='utf-8') -> str:
    """将传入的字符串转换为md5字符串"""
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode('utf-8'))
    return md5_hash.hexdigest()

class KnowledgeBaseService(object):
    """知识库服务类"""
    
    def __init__(self):
        # r如果文件夹不存在，则创建;存在就跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        
        self.chroma = Chroma(
            collection_name=config.collection_name,  # 数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,  # 数据库本地存储文件夹
        )  # 向量存储的实例 Chroma向量库对象
        
        # 初始化文本分割器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 每个文本块的大小
            chunk_overlap=config.chunk_overlap,  # 文本块之间的重叠大小
            separators=config.separators,  # 分隔符列表
            length_function=len,  # 计算文本长度的函数
        )  # 文本分割器的对象
    
    def upload_by_str(self, data, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        # 计算文件内容的 MD5
        file_md5 = get_string_md5(data)
        
        # 检查是否已处理过
        if check_md5(file_md5):
            return {"status": "success", "message": "文件已存在，无需重复上传"}
        
        # 使用分割器分割文本
        docs = self.splitter.split_text(data)
        
        # 为每个文本块添加元数据
        metadata = {"source": filename, "author": "liusaidh"}
        
        # 存入向量数据库
        self.chroma.add_texts(
            texts=docs,
            metadatas=[metadata for _ in docs]
        )
        
        # 保存 MD5
        save_md5(file_md5)
        
        return {"status": "success", "message": f"文件 {filename} 上传成功，共 {len(docs)} 个文本块"}

# 测试代码：
if __name__ == '__main__':
    # 测试 get_string_md5
    test_text = "这是一个测试文本1"
    test_text1 = "这是一个测试文本"

    md5_result = get_string_md5(test_text)
    md5_result1 = get_string_md5(test_text1)
    print(f"原始文本: {test_text}")
    print(f"MD5值: {md5_result}")
    print(f"原始文本: {test_text1}")
    print(f"MD5值: {md5_result1}")


    service = KnowledgeBaseService()
    print(service.upload_by_str("王者荣耀", "testfile"))
    

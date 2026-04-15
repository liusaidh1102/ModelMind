from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# Chroma 向量存储 ,向量数据库（轻量级的）


# 1. 初始化内存向量存储，绑定通义千问DashScope嵌入模型
# 内存存储
vector_store = Chroma(
    collection_name="my_collection_test", # 当前向量数据库起个名字，类似表名
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma_db" # 存储数据的目录

)

# 2. 加载CSV文件
loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source",  # 指定本条数据的来源字段
)

# 3. 加载CSV为LangChain Document对象列表
documents = loader.load()

# 4. 向向量存储添加文档，并自动生成递增ID（id1, id2, id3...）
vector_store.add_documents(
    documents=documents,  # 被添加的文档，类型：list[Document]
    ids=["id"+str(i) for i in range(1, len(documents)+1)]  # 给每个文档分配唯一字符串ID
)

# 5. 删除指定ID的文档
vector_store.delete(["id1", "id2"])

# 6. 执行相似性检索，返回Top-3最相关的文档
result = vector_store.similarity_search(
    query="Python是不是简单易学呀",  # 用户查询文本
    k=3,  # 返回最相关的3个结果
    filter={
        "source": "Python官方教程"  # 结果的过滤
    }
)

print( result)
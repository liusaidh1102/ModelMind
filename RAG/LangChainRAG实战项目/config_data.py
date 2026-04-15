

# 定义一些变量
md5_path = "./md5.txt"
collection_name = "rag"
persist_directory = "./chroma_db"

# 文本分割配置
chunk_size = 500  # 每个文本块的大小
chunk_overlap = 50  # 文本块之间的重叠大小
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " "]

# 向量数据库返回的结果数量
similarity_threshold = 3

embeeding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {"configurable": {"session_id": "user_001"}}
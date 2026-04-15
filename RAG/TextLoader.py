from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载本地文本文件
loader = TextLoader(
    "./data/超长测试文本.txt",
    encoding="utf-8",
)
docs = loader.load() # 获取一个文档


# 初始化递归字符文本分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 分段的最大字符数
    chunk_overlap=50,      # 分段之间允许重叠的字符数
    # 文本分段依据（按优先级从高到低尝试分割）
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    # 字符统计依据（函数）
    length_function=len,
)

# 执行文档分割
split_docs = splitter.split_documents(docs)
print(len(split_docs))
print(split_docs)
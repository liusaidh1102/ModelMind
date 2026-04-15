from langchain_community.document_loaders import PyPDFLoader

# 初始化PDF加载器
loader = PyPDFLoader(
    file_path="./data/（备份）深入理解Java虚拟机：JVM高级特性与最佳实践（第3版） 【文字版】 (周志明 [周志明]) (Z-Library).pdf",  # 【必填】PDF文件的本地路径/绝对路径
    # mode="page",   # 读取模式：可选 page / single
    # password="password"  # 【可选】PDF加密文件的访问密码
)
docs = loader.load()
for doc in loader.lazy_load():
    print(doc)

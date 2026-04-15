from langchain_community.document_loaders import JSONLoader

# # 初始化JSONLoader
# loader = JSONLoader(
#     file_path="./data/student.json",  # JSON文件路径
#     jq_schema=".other",           # jq语法：提取other字段
#     text_content = False  # 加上这一行！允许 page_content 是字典
# )
#
# # 加载文档
# document = loader.load()
# # 打印结果
# print(document)

# 初始化JSONLoader
loader = JSONLoader(
    file_path="./data/student_json_lines.json",  # JSON文件路径
    jq_schema=".",           # jq语法：提取other字段
    text_content = False,  # 加上这一行！允许 page_content 是字典
    json_lines = True # 允许 JSON 文件是 JSON Lines 格式
)

# 加载文档
document = loader.load()
# 打印结果
print(document)
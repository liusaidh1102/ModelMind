"""
提示词：用户的提问 + 向量库中检索到的参考资料
"""
from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化通义千问大模型（使用qwen3-max最强版本）
model = ChatTongyi(model="qwen3-max")

# 2. 定义提示词模板（System + User 双轮结构）
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问: {input}")
    ]
)

# 3. 初始化内存向量存储，指定通义文本嵌入模型v4
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 4. 准备资料（向量库的数据）
# add_texts 传入一个 list[str]，直接传入纯文本字符串   ，一般是文件转换为document，然后再传入向量数据库
vector_store.add_texts(
    [
        "减肥就是要少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
        "跑步是很好的运动哦"
    ]
)

# 5. 用户查询
input_text = "怎么减肥？"

# 6. 检索向量库：返回Top-2条最相关的文档
result = vector_store.similarity_search(input_text, k=2)
reference_txt = "["
# 7. 打印检索结果
for doc in result:
    reference_txt += doc.page_content
reference_txt += "]"

print(reference_txt)

# 打印模型的输出内容
def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt


# 构建chain
chain = prompt | print_prompt | model | StrOutputParser()

res = chain.invoke({"input" : input_text,"context": reference_txt})
print( res)
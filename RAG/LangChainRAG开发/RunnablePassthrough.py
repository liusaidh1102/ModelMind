"""
提示词：用户的提问 + 向量库中检索到的参考资料 + 通过RunnablePassthrough让向量数据库入链
"""
from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# retriever作为向量数据库的检索结果， langchain中向量存储对象，有一个方法：as_retriever，可以返回一个Runnable接口的子类实例对象
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

'''
初步链条：
chain = retriever | prompt | model | StrOutputParser()
但是分析输入和输出：
retriever:
 - 输入：用户的提问 str
 - 输出：向量数据库的检索结果 list[Document]
prompt:
    - 输入：用户的提问 + 向量数据库的检索结果 dict
    - 输出：完整的提示词  PromptValue
结果：retriever的输出结果不能作为prompt的输入，因为他要的是dict字典，同时用户的提问会丢失

要改写链：
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
)
res = chain.invoke(input_text)
整个字典里的所有 value，都会同时、自动收到 invoke(...) 里的内容！
1.RunnablePassthrough()作用： 接收输入 → 原样输出
2.用户的输入还会给retriever

'''




# 将检索结果List[Document]格式化成字符串
def format_func(docs):
    if not docs:
        return "无参考资料"
    return "参考资料：" + "\n".join([doc.page_content for doc in docs])

# 打印模型的输出内容
def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# chain
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
)
'''
'''
res = chain.invoke(input_text)
print(res)
# # langchain_community
# from langchain_community.llms.tongyi import Tongyi
#
# # 不用qwen3-max，因为qwen3-max是聊天模型，qwen-max是大语言模型
# model = Tongyi(model="qwen-max")
#
# # 调用invoke向模型提问
# res = model.invoke(input="你是谁呀能做什么？")
#
# print(res)
#
#
#
# # langchain_ollama
# from langchain_ollama import OllamaLLM
#
# model = OllamaLLM(model="qwen3:4b")
#
# res1 = model.invoke(input="你是谁呀能做什么？")
#
# print(res1)


# # langchain_ollama 流式输出
# from langchain_ollama import OllamaLLM
#
# model = OllamaLLM(model="qwen3:4b")
#
# # 流式输出
# for chunk in model.stream("你是谁呀能做什么？"):
#     print(chunk, end="", flush=True)

# # chat_model
# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
#
# # 初始化模型
# chat = ChatTongyi(model="qwen3-max")
#
# # 准备消息list
# messages = [
#     SystemMessage(content="你是一名来自边塞的诗人"),
#     HumanMessage(content="给我写一首唐诗"),
#     AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
#     HumanMessage(content="给予你上一首的格式，再来一首")
# ]
#
# # 流式输出
# for chunk in chat.stream(input=messages):
#     print(chunk.content, end="", flush=True)

# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 得到模型对象，qwen3-max就是聊天模型
# model = ChatTongyi(model="qwen3-max")
#
# # 准备消息列表（元组格式）
# type = "边塞" # 可以支持变量嵌入
# messages = [
#     ("system", "你是一个边塞诗人。"),
#     ("human", "写一首唐诗"),
#     ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
#     ("human", "按照你上一个回复的格式，再写一首唐诗，关于{type}的。")
# ]
#
# # for循环迭代打印输出，通过.content来获取到内容
# for chunk in model.stream(input=messages):
#     print(chunk.content, end="", flush=True)


# from langchain_community.embeddings import DashScopeEmbeddings
#
# # 初始化嵌入模型对象，其默认使用模型是：text-embedding-v1
# embed = DashScopeEmbeddings()
#
# # 测试
# print(embed.embed_query("我喜欢你"))
# print(embed.embed_documents(['我喜欢你', '我稀饭你', '晚上吃啥']))


# LangChain的提示词模版
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms.tongyi import Tongyi
#
# # 构建提示词模板
# prompt_template = PromptTemplate.from_template(
#     "我的邻居姓{lastname}，刚生了{gender}，帮忙起名字，请简略回答。"
# )
#
# # 变量注入，生成提示词文本
# prompt_text = prompt_template.format(lastname="张", gender="女儿")
#
# # 创建模型对象
# model = Tongyi(model="qwen-max")
# # 调用模型获取结果
# res = model.invoke(input=prompt_text)
# print(res)

# 基于chain链的写法
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms.tongyi import Tongyi
#
# # 构建提示词模板
# prompt_template = PromptTemplate.from_template(
#     "我的邻居姓{lastname}，刚生了{gender}，帮忙起名字，请简略回答。"
# )
#
# # 创建模型对象
# model = Tongyi(model="qwen-max")
# # 生成链：将模板与模型串联
# chain = prompt_template | model
#
# # 基于链，直接传入参数调用模型获取结果
# res = chain.invoke(input={"lastname": "曹", "gender": "女儿"})
# print(res)

# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain_community.llms.tongyi import Tongyi
# # 定义单个示例的模板
# example_template = PromptTemplate.from_template("单词:{word}, 反义词:{antonym}")
#
# # 示例数据：list 内套字典，用于给模型做Few-Shot学习
# example_data = [
#     {"word": "大", "antonym": "小"},
#     {"word": "上", "antonym": "下"}
# ]
#
# # 组装FewShotPromptTemplate对象
# few_shot_prompt = FewShotPromptTemplate(
#     example_prompt=example_template,  # 单个示例的渲染模板
#     examples=example_data,            # 示例数据集
#     prefix="给出给定词的反义词，有如下示例：",  # 示例前的引导语
#     suffix="基于示例告诉我：{input_word}的反义词是？",  # 示例后的用户问题
#     input_variables=['input_word']    # 最终需要传入的变量
# )
#
# # 调用模板，传入变量，生成最终提示词
# prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()
# print(prompt_text)
#
# model = Tongyi(model="qwen-max")
# res = model.invoke(input=prompt_text)
# print(res)



# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 构建对话式提示词模板
# chat_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个边塞诗人，可以作诗。"),
#         MessagesPlaceholder("history"),  # 对话历史占位符
#         ("human", "请再来一首唐诗"),
#     ]
# )
#
# # 对话历史数据：多轮 human/ai 对话
# history_data = [
#     ("human", "你来写一个唐诗"),
#     ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
#     ("human", "好诗再来一个"),
#     ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
# ]
#
# # 注入对话历史，生成最终提示词
# prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
#
# # 初始化通义千问对话模型
# model = ChatTongyi(model="qwen3-max")
#
# # 调用模型获取结果
# res = model.invoke(prompt_text)
#
# # 打印回答内容和类型
# print(res.content, type(res))



# # chain的使用
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 构建对话式提示词模板
# chat_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个边塞诗人，可以作诗。"),
#         MessagesPlaceholder("history"),  # 对话历史占位符
#         ("human", "请再来一首唐诗"),
#     ]
# )
#
# # 对话历史数据：多轮 human/ai 对话
# history_data = [
#     ("human", "你来写一个唐诗"),
#     ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
#     ("human", "好诗再来一个"),
#     ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
# ]
#
# # 注入对话历史，生成最终提示词
# prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
#
# # 初始化通义千问对话模型
# model = ChatTongyi(model="qwen3-max")
#
# # 普通写法
# # res = model.invoke(prompt_text)
# # print(res.content, type(res))
# # 链式写法
# chain = chat_prompt_template | model
# # 通过链调用模型获取结果，通过invoke或stream方法获取结果
# res = chain.invoke(input={"history": history_data})
# print(res.content, type(res))


# # StrOutputParser 是 LangChain 中用于处理模型输出类型转换的核心解析器，本质是一个 字符串转换器。
# # 为了：将 AIMessage 转为字符串，解决  chain = prompt | model | model  报错的问题
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 1. 定义模板
# prompt = ChatPromptTemplate.from_template("介绍一下{topic}")
#
# # 2. 定义模型
# model = ChatTongyi(model="qwen-max")
#
# # 3. 定义解析器（核心：将AIMessage转为字符串）
# parser = StrOutputParser()
#
# # 4. 构建管道链：模板 → 模型 → 解析器 → 后续处理
# # 链的最终输出类型是 str，而不是 AIMessage
# chain = prompt | model | parser | model
#
# # 5. 调用
# result = chain.invoke({"topic": "程序员"})
# print(type(result))  # 输出: <class 'AiMessage'>
# print(result.content)        # 输出: 程序员是从事...的专业人员

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 初始化解析器
# str_parser = StrOutputParser()
# json_parser = JsonOutputParser()
#
# # 初始化通义千问模型
# model = ChatTongyi(model="qwen3-max")
#
# # 第一轮提示词模板：生成JSON格式的名字
# first_prompt = PromptTemplate.from_template(
#     "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装到JSON格式返回给我，"
#     "要求封装为json格式。key是name，value就是起的名字。请严格遵守格式要求"  # 这里要指定对应的要求，转换为key，value格式，便于后面的转换
# )
#
# # 第二轮提示词模板：解析名字含义
# second_prompt = PromptTemplate.from_template(
#     "姓名{name}，请帮我解析含义。"
# )
#
# # 构建完整LCEL链
# chain = first_prompt | model | json_parser | second_prompt | model | str_parser
#
# # 调用链并获取结果
# res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
# print(res)
# print(type(res))


# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda
# from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models.tongyi import ChatTongyi
#
# # 初始化字符串解析器
# str_parser = StrOutputParser()
#
# # 自定义RunnableLambda：将模型返回的AIMessage转为字典，提取name字段
# my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})
#
# # 初始化通义千问模型
# model = ChatTongyi(model="qwen3-max")
#
# # 第一轮提示词模板：仅要求返回名字
# first_prompt = PromptTemplate.from_template(
#     "我邻居姓: {lastname}，刚生了{gender}，请起名，仅告知我名字，不要额外信息"
# )
#
# # 第二轮提示词模板：解析名字含义
# second_prompt = PromptTemplate.from_template(
#     "姓名{name}，请帮我解析含义。"
# )
#
# # 构建完整LCEL链
# chain = first_prompt | model | my_func | second_prompt | model | str_parser
#
# # 调用链并获取结果
# res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
# print(res)
# print(type(res))


# 短期会话记忆
# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables.history import RunnableWithMessageHistory
#
# def print_prompt(full_prompt):
#     print("="*20, full_prompt.to_string(), "="*20)
#     return full_prompt
#
# model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template(
#     "你需要根据对话历史回应用户问题。对话历史：{chat_history}。用户当前输入：{input}， 请给出回应"
# )
# base_chain = prompt | print_prompt | model | StrOutputParser()
#
# chat_history_store = {}  # 存放多个会话ID所对应的历史会话记录
# def get_history(session_id):
#     if session_id not in chat_history_store:
#         # 存入新的实例
#         chat_history_store[session_id] = InMemoryChatMessageHistory()
#     return chat_history_store[session_id]
#
# # 通过RunnableWithMessageHistory获取一个新的带有历史记录功能的chain
# conversation_chain = RunnableWithMessageHistory(
#     base_chain,  # 被附加历史消息的Runnable，通常是chain
#     get_history,  # 获取历史会话的函数
#     input_messages_key="input",  # 声明用户输入消息在模板中的占位符
#     history_messages_key="chat_history"  # 声明历史消息在模板中的占位符
# )
#
# if __name__ == '__main__':
#     # 如下固定格式，配置当前会话的ID
#     session_config = {"configurable": {"session_id": "user_001"}}
#
#     print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
#     print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
#     print(conversation_chain.invoke({"input": "一共有几个动物？"}, session_config))


# # 基于文件的长期会话记忆
# import json
# import os
# from typing import Sequence
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
#
#
# class FileChatMessageHistory(BaseChatMessageHistory):
#     storage_path: str
#     session_id: str
#
#     def __init__(self, storage_path: str, session_id: str):
#         self.storage_path = storage_path
#         self.session_id = session_id
#
#     @property  # @property装饰器，将方法变为属性，方便后续通过对象.属性访问
#
#     def messages(self) -> list[BaseMessage]:
#         file_path = os.path.join(self.storage_path, self.session_id)
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 messages_data = json.load(f)
#             return messages_from_dict(messages_data)
#         except FileNotFoundError:
#             return []
#
#     def add_messages(self, messages: Sequence[BaseMessage]) -> None:
#         # 获取现有消息 + 追加新消息
#         all_messages = list(self.messages) # self.messages获取父类的属性，里面是全部的消息
#         all_messages.extend(messages)
#
#         # 序列化消息为字典格式
#         serialized = [message_to_dict(message) for message in all_messages]
#         file_path = os.path.join(self.storage_path, self.session_id)
#
#         # 确保目录存在，写入JSON文件
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(serialized, f, ensure_ascii=False, indent=2)
#
#     def clear(self) -> None:
#         file_path = os.path.join(self.storage_path, self.session_id)
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump([], f)
#
# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables.history import RunnableWithMessageHistory
#
# def print_prompt(full_prompt):
#     print("="*20, full_prompt.to_string(), "="*20)
#     return full_prompt
#
# model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template(
#     "你需要根据对话历史回应用户问题。对话历史：{chat_history}。用户当前输入：{input}， 请给出回应"
# )
# base_chain = prompt | print_prompt | model | StrOutputParser()
#
# # 换成文件存储
# def get_history(session_id):
#     return FileChatMessageHistory(storage_path="./chat_history", session_id=session_id)
#
#
# # 通过RunnableWithMessageHistory获取一个新的带有历史记录功能的chain
# conversation_chain = RunnableWithMessageHistory(
#     base_chain,  # 被附加历史消息的Runnable，通常是chain
#     get_history,  # 获取历史会话的函数
#     input_messages_key="input",  # 声明用户输入消息在模板中的占位符
#     history_messages_key="chat_history"  # 声明历史消息在模板中的占位符
# )
#
# if __name__ == '__main__':
#     # 如下固定格式，配置当前会话的ID
#     session_config = {"configurable": {"session_id": "user_001"}}
#
#     # print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
#     # print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
#     print(conversation_chain.invoke({"input": "一共有几个动物？"}, session_config))



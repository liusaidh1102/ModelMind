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


# langchain_ollama 流式输出
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

# 流式输出
for chunk in model.stream("你是谁呀能做什么？"):
    print(chunk, end="", flush=True)
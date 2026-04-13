from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    # system角色：全局设定
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    # assistant角色：历史回复
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    # user角色：用户提问
    {"role": "user", "content": "1+1等于几？"},
    # 👇 下面就是【会话记忆】，把AI的回答加上
    {"role": "assistant", "content": "1+1等于2。"},
    # 👇 新提问，能记住上下文
    {"role": "user", "content": "再加3等于几？"}
]
response = client.chat.completions.create(
    model="qwen3.5-plus",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    stream=True  # 开启流式返回
)

for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content,
              end="",  # 每一段用""分割
              flush=True  # 立刻刷新缓冲区
              )


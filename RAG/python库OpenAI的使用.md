# OpenAI库

## 1. 基本使用

| 角色        | 作用         | 特点                                                         |
| ----------- | ------------ | ------------------------------------------------------------ |
| `system`    | 全局系统提示 | 设定 AI 助手的身份、行为规则、回答风格，是全局背景，影响所有后续交互 |
| `assistant` | AI 助手      | 代表模型的历史回复，用于维护多轮对话上下文，第一次是对system的回复 |
| `user`      | 用户         | 代表用户的提问、指令或需求                                   |

```python
from openai.types.chat.chat_completion import ChatCompletion

# 调用模型生成对话
response: ChatCompletion = client.chat.completions.create(
    model="qwen3-max",  # 指定模型
    messages=[
        # system角色：全局设定
        {"role": "system", "content": "你是一个Python编程专家。"},
        # assistant角色：历史回复
        {"role": "assistant", "content": "我是一个Python编程专家。请问有什么可以帮助您的吗？"},
        # user角色：用户提问
        {"role": "user", "content": "for循环输出1到5的数字"}
    ]
)
```

response作为响应结果如下：可以通过print(response.choices[0].message.content)获取响应的结果。

```json

{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1735689600,
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "生成的回复内容"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 80,
    "total_tokens": 130
  }
}
```

完整代码：

```python
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    # system角色：全局设定
    {"role": "system", "content": "你是一个Python编程专家。"},
    # assistant角色：历史回复
    {"role": "assistant", "content": "我是一个Python编程专家。请问有什么可以帮助您的吗？"},
    # user角色：用户提问
    {"role": "user", "content": "for循环输出1到5的数字"}
]
response = client.chat.completions.create(
    model="qwen3.5-plus",  # 您可以按需更换为其它深度思考模型
    messages=messages,
)
print(response.choices[0].message.content)
```

## 2. 流式输出

```python
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    # system角色：全局设定
    {"role": "system", "content": "你是一个Python编程专家。"},
    # assistant角色：历史回复
    {"role": "assistant", "content": "我是一个Python编程专家。请问有什么可以帮助您的吗？"},
    # user角色：用户提问
    {"role": "user", "content": "for循环输出1到5的数字"}
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
```

## 3. 会话记忆

把会话记忆存在对应的List里面就行了。

```python
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
```
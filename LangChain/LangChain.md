# LangChain

## 1. 简介

LangChain 是一款**面向大语言模型（LLM）业务开发的 Python 第三方库**，核心定位是 “LLM 业务功能集大成者”，通过封装标准化 API，降低 LLM 应用开发的复杂度，提供一站式的功能集成与流程编排能力。

提供的功能：提示词优化、模型调用（支持各种模型）、会话记忆、文档管理分析、Agent 智能体构建、链式执行。

## 2. 安装对应的包

```python
pip install langchain langchain-community langchain-ollama dashscope chromadb
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain langchain-community langchain-ollama dashscope chromadb # 添加国镜像源下载更快。
```

- **langchain**：LangChain 核心基础包，提供Chain、提示词、记忆、Agent等核心功能 ;

- **langchain-community**：社区扩展包，集成第三方大模型、向量库、工具组件; 
- **langchain-ollama**：专门对接本地Ollama模型，支持离线调用开源大模型 
- **dashscope**：阿里云通义千问官方SDK，用于调用千问大模型 
- **chromadb**：轻量嵌入式向量数据库，用于RAG场景存储和检索文本向量


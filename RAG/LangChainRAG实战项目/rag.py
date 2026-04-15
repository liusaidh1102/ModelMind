from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):
    def __init__(self):
        # 初始化向量库服务
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embeeding_model_name)
        )

        # 提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}。"),
                ("system","并且我提供用户的对话历史记录，如下"),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        # 大模型
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 构建 RAG 链
        self.chain = self.__get_chain()

    def __get_chain(self):
        # 获取检索器
        retriever = self.vector_service.get_retriever()

        # 文档格式化函数
        def format_docs(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n"
            return formatted_str

        def format_for_retriever(value:dict) -> str:
            # print("---------------------",value)
            # {'input': 'redis常见的数据类型？', 'history': []}
            return value['input'] # 这里会把history给丢掉，就返回了input

        def format_for_prompt(value):
            # 这里要返回 {input,history,context}
            new_value = {
                "input": value["input"]["input"],
                "history": value["input"]["history"],
                "context": value["context"]
            }
            # print("---------------------",value)
            return new_value

        # 构建完整链
        chain = (
                {
                    "input": RunnablePassthrough(),
                    # {'input': 'redis常见的数据类型？', 'history': []} 交给了retriever，但是retriever要的是一个字符串input
                    "context": RunnableLambda(format_for_retriever) | retriever | format_docs
                } |
                RunnableLambda(format_for_prompt)
                | self.prompt_template
                | print_prompt
                | self.chat_model
                | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return conversation_chain


if __name__ == '__main__':
    # session_id配置
    session_config = {"configurable": {"session_id": "user_001"}}
    service = RagService()
    # 增强链的输入，要求是一个dict
    res = service.chain.invoke({"input": "redis的string数据类型？"}, session_config)
    print(res)
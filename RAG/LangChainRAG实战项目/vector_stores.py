import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

'''
作用：
返回检索器用于加入链
'''
class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": config.similarity_threshold
            }
        )
if __name__ == '__main__':
    service = VectorStoreService(DashScopeEmbeddings(model = config.embeeding_model_name))
    retriever = service.get_retriever()
    res =  retriever.invoke("redis常见的数据类型？")
    print(res)

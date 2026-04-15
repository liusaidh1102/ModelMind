# 基于文件的长期会话记忆
import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    storage_path: str
    session_id: str

    def __init__(self, storage_path: str, session_id: str):
        self.storage_path = storage_path
        self.session_id = session_id

    @property  # @property装饰器，将方法变为属性，方便后续通过对象.属性访问

    def messages(self) -> list[BaseMessage]:
        file_path = os.path.join(self.storage_path, self.session_id)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # 获取现有消息 + 追加新消息
        all_messages = list(self.messages) # self.messages获取父类的属性，里面是全部的消息
        all_messages.extend(messages)

        # 序列化消息为字典格式
        serialized = [message_to_dict(message) for message in all_messages]
        file_path = os.path.join(self.storage_path, self.session_id)

        # 确保目录存在，写入JSON文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)

# 换成文件存储
def get_history(session_id):
    return FileChatMessageHistory(storage_path="./chat_history", session_id=session_id)



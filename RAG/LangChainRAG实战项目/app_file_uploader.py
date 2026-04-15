# 基于 Streamlit 的 Web 应用程序
"""
    stream特点：当页面的元素发生更改，代码重头开始跑一遍，提供了session_state
    功能描述：上传TXT文件并展示文件内容,同时调用knowledge_base中的方法，去上传到向量库中去
    pip install streamlit
    运行命令：streamlit run app_file_uploader.py
"""
import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
uploader_file = st.file_uploader(
    label="请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,  # False表示仅接受一个文件的上传
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB

    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")

    # get_value -> bytes -> decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8")
    st.write(text)
    with st.spinner("文件处理中..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)


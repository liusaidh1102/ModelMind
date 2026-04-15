import time

import streamlit as st
from rag import RagService
import config_data as config

# 标题
st.title("智能客服")
st.divider()


if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，请问有什么能帮助你的？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])


# 在页面的最下方显示输入框
prompt = st.chat_input()


if prompt:

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []

    # 获取AI回复
    with st.spinner("AI思考中..."):
        time.sleep(1)
        # 询问ai
        response_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        # 显示AI回复

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(response_stream, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})


import streamlit as st
import time
from model import qwen15_05b_chat as model


class ChatRecord:
    def __init__(self, avatar: str, role: str, content: str):
        self.avatar = avatar
        self.role = role
        self.content = content


def ai_stream_data(content: str, intervals: float = 0):
    for word in content:
        yield word
        time.sleep(intervals)


def on_submit():
    pass


def page():
    st.title(f"你好 {st.session_state.login_username}，欢迎使用Chat AI。")

    if len(st.session_state.chat_records) == 0:
        aiChatRecord = ChatRecord("ai", "system", f"你好 {st.session_state.login_username}，可以开始提问了哦。")
        with st.chat_message("ai"):
            st.write_stream(ai_stream_data(f"{aiChatRecord.content}", 0.1))
        st.session_state.chat_records.append(aiChatRecord)
    else:
        for chat_record in st.session_state.chat_records:
            with st.chat_message(chat_record.avatar):
                st.write_stream(ai_stream_data(f"{chat_record.content}"))

    if "user_chat_input" in st.session_state and st.session_state.user_chat_input is not None:
        # AI回答时展示一个不可输入的输入框
        st.chat_input("AI正在组织语言回复中，稍后可以继续提问哦...", disabled=True)

        userChatRecord = ChatRecord("user", "user", st.session_state.user_chat_input)
        st.session_state.chat_records.append(userChatRecord)

        # 调用模型
        modelOutputIter = model.INS.generate(st.session_state.chat_records)
        with st.chat_message('user'):
            st.write_stream(ai_stream_data(f"{userChatRecord.content}"))
        with st.chat_message('ai'):
            st.write_stream(modelOutputIter)

        aiChatRecord = ChatRecord("ai", "assistant", modelOutputIter.output)
        st.session_state.chat_records.append(aiChatRecord)

        # 刷新页面
        st.rerun()

    st.chat_input("请输入...", key='user_chat_input', on_submit=on_submit)


def chat_ai_page():
    page()

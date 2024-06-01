import streamlit as st

def page():
    st.warning("你还没有登录，请先登录哦。")

    with st.form(key='login'):
        username = st.text_input(label='用户名')
        password = st.text_input(label='密码', type='password')
        is_submitted = st.form_submit_button(label='登录')
        if is_submitted:
            st.session_state.login_username = username
            # 重新加载
            st.rerun()
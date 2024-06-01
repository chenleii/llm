import streamlit as st
from streamlit_option_menu import option_menu
import ui.pages as pages
from ui.pages.about import about_page
from ui.pages.chat_ai import chat_ai_page


def session_state_init():
    if "login_username" not in st.session_state:
        # st.session_state.login_username = None
        st.session_state.login_username = 'cc'

    if "chat_records" not in st.session_state:
        st.session_state.chat_records = []


def init():
    session_state_init()

    # 检查登录
    if st.session_state.login_username is None:
        pages.login.page()
        return

    with st.sidebar:
        st.caption(
            f"""<p align="right">当前版本：{'1.0.0'}</p>""",
            unsafe_allow_html=True,
        )

    with st.sidebar:
        menus = {
            "Chat AI": {
                "icon": "chat",
                # "page_func": pages.chat_ai.page,
                "page_func": chat_ai_page,
            },
            "About": {
                "icon": "hdd-stack",
                # "page_func": pages.about.page,
                "page_func": about_page,
            },
        }
        menu_options = list(menus.keys())
        menu_icons = [x["icon"] for x in menus.values()]
        selected_page = option_menu(
            "",
            options=menu_options,
            icons=menu_icons,
            default_index=0,
        )

    if selected_page in menus:
        menus[selected_page]["page_func"]()



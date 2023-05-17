import asyncio
import streamlit as st
from blog_summarize import summarize_blog

st.set_page_config(page_title="Blog Summarizer", page_icon="📝")
st.markdown("""
# 网页总结器
**这是一个把网页内容总成简单文字的工具。**
""")

st.text_input("第一步，输入想总结的网页地址", key="url")
st.selectbox("第二步，选择总结用的语言", ["Chinese", "English"], key="lang")


async def _sum(url, lang):
    if url and lang:
        st.write("正在总结，请稍等...")
        summary = st.empty()
        result = summarize_blog(url=url, lang=lang)
        summary.write(result)
    else:
        st.write("请先输入网页地址和选择语言。")

if st.button("总结"):
    print("总结")
    url = st.session_state.url
    lang = st.session_state.lang
    asyncio.run(_sum(url, lang))

import streamlit as st

# -------------------------- 页面配置（GPT风格）--------------------------
st.set_page_config(page_title="GPT图文对话", layout="wide")
st.title("💬 GPT 图文助手")

# -------------------------- 初始化聊天记忆 --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------- 渲染所有聊天记录 --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 先显示图片，再显示文字（和GPT完全一致）
        if "image" in msg and msg["image"] is not None:
            st.image(msg["image"], width=400)
        st.markdown(msg["content"])

# -------------------------- GPT 同款上传图片+输入框 --------------------------
# 分栏布局：左边上传图片，右边输入文字（完美模仿GPT）
col1, col2 = st.columns([1, 4])
with col1:
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
with col2:
    prompt = st.chat_input("输入你的问题...")

# -------------------------- 处理用户消息 --------------------------
if prompt:
    # 1. 展示并保存用户消息（文字+图片）
    with st.chat_message("user"):
        if uploaded_file:
            st.image(uploaded_file, width=400)
        st.markdown(prompt)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image": uploaded_file
    })

    # 2. AI 回复（可后续对接GPT/多模态模型）
    ai_reply = "✅ 已收到你的图片和问题！我可以帮你分析图片、回答问题～"
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply,
        "image": None
    })
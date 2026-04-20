# =========================
# 标准库导入
# =========================
import warnings
import logging
import os

# =========================
# 第三方库导入
# =========================
import torch
import streamlit as st
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

# =========================
# 项目配置导入
# =========================
from config import *

# =========================
# 环境配置
# =========================
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="强直性脊柱炎智能诊疗助手",
    layout="wide"
)

# =========================
# 🔥 核心修复：全新CSS布局（GPT风格，永不重叠）
# =========================
st.markdown("""
<style>
/* 1. 全局基础：隐藏默认间距，优化聊天样式 */
.block-container {
    padding-bottom: 100px !important; /* 给底部栏留空间，核心！ */
    padding-top: 2rem !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 18px;
    line-height: 1.6;
}
textarea {
    font-size: 18px !important;
}

/* 2. 底部固定容器：左侧上传 + 右侧输入框（Flex弹性布局） */
div.fixed-bottom-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #0e1117;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 15px; /* 按钮和输入框的间距 */
    z-index: 9999;
    border-top: 1px solid #262730;
}

/* 3. 上传按钮样式：固定宽度，美观大气 */
div[data-testid="stFileUploader"] {
    width: 220px !important;
    margin: 0 !important;
}
div[data-testid="stFileUploader"] button {
    font-size: 16px !important;
    padding: 8px 14px !important;
    width: 100% !important;
}
div[data-testid="stFileUploader"] small {
    font-size: 12px !important;
    white-space: normal !important;
}

/* 4. 聊天输入框：占满剩余宽度，完美适配 */
section[data-testid="stChatInput"] {
    flex: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>

<!-- 自定义底部容器，包裹上传按钮 + 输入框 -->
<div class="fixed-bottom-bar">
""", unsafe_allow_html=True)

# =========================
# Session 初始化
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# =========================
# 加载模型 + RAG
# =========================
@st.cache_resource
def load_model_and_rag():
    # 加载Qwen2VL视觉语言模型
    # BASE_MODEL_PATH: 模型文件路径
    # torch_dtype=torch.bfloat16: 使用bfloat16精度加速推理
    # device_map="auto": 自动分配设备（CPU/GPU）
    # trust_remote_code=True: 信任远程代码（用于加载自定义模型）
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载模型处理器，负责文本和图像的预处理
    # processor用于将输入转换为模型可接受的格式
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    # 加载文本嵌入模型，用于RAG中的语义检索
    # model_name: 使用中文文本嵌入模型
    # model_kwargs={'device': "cuda"}: 使用GPU加速嵌入计算
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': "cuda"}
    )

    # 加载Chroma向量数据库，用于存储和检索专业知识
    # persist_directory: 数据库持久化存储路径
    # embedding_function: 使用上面定义的嵌入函数
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    # 返回加载好的模型、处理器和向量数据库
    return model, processor, db

# 检查模型是否已加载，避免重复加载
if not st.session_state.model_loaded:
    # 显示加载状态提示，提升用户体验
    with st.spinner("🚀 正在加载模型和专业知识库..."):
        # 调用模型和RAG加载函数
        model, processor, db = load_model_and_rag()
        # 将加载的模型、处理器和向量数据库存储到session_state中
        # 这样可以在整个会话中重复使用，避免重复加载
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.db = db
        # 标记模型已加载状态
        st.session_state.model_loaded = True

# =========================
# 标题
# =========================
st.title("强直性脊柱炎智能诊疗助手")

# =========================
# 聊天区
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=150)
        st.markdown(msg["content"])

# =========================
# 关键：上传按钮 + 输入框 放入底部Flex容器
# =========================
# 图片上传（左侧固定）
uploaded_file = st.file_uploader(
    "上传影像图片",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}"
)

uploaded_image = None
if uploaded_file:
    uploaded_image = Image.open(uploaded_file)

# 聊天输入框（右侧占满）
prompt = st.chat_input("请输入问题（Enter发送）")

# 关闭底部容器
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 推理函数（RAG链路+提示词深度优化，其余逻辑原样保留）
# =========================
def get_answer(text, img):
    # 从session_state中获取预加载的模型、处理器和向量数据库
    model = st.session_state.model
    processor = st.session_state.processor
    db = st.session_state.db

    # 初始化图片描述，默认为"无图片"
    img_desc = "无图片"

    # Step1: 图片理解 - 原图提示词完全不动
    if img:
        image = img.convert("RGB")
        prompt_img = """你是风湿免疫科放射专科医师，严格客观描述骶髂关节强直性脊柱炎影像病灶：
1. 骶髂X光平片仅描述：骨质侵蚀、骨质硬化、关节间隙宽窄、骨性融合、韧带钙化
2. 骶髂MRI核磁仅描述：骨髓水肿、肌腱附着点炎症、滑膜炎性改变
3. 严禁跨影像编造征象：X光绝不描述骨髓水肿，不混用核磁病理特征
4. 只描述本次病变，不脑补脊柱、其他部位额外病灶
5. 术语专业简洁，不描述人物、背景无关内容，不结合症状推断病情分期"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_img}
            ]
        }]
        inputs = processor(
            text=processor.apply_chat_template(messages, add_generation_prompt=True),
            images=image,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, repetition_penalty=1.25)
        img_desc = processor.decode(out[0], skip_special_tokens=True)
        if "assistant" in img_desc:
            img_desc = img_desc.split("assistant")[-1]

    # ==============================================
    # ========== 【优化1】全新规范RAG检索链路 ==========
    # 适配：单维度短句 + 完整五维联合病例，去重+多维度交叉召回
    # ==============================================
    full_query = f"骶髂关节影像表现：{img_desc}；患者临床症状：{text}"

    # 扩大召回数量，兼顾单项匹配+完整对照病例，医疗最优检索
    retriever = db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(full_query)
    
    # 内容去重，避免重复冗余参考
    unique_doc_map = {doc.page_content: doc for doc in docs}
    unique_docs = list(unique_doc_map.values())
    
    # 选取最优4条上下文，不超长、不杂乱
    context = "\n".join([d.page_content for d in unique_docs[:4]])

    # ==============================================
    # ========== 【优化2】终极防幻觉临床诊疗提示词 ==========
    # 严格遵循多维度联合诊断，绝不编造用户不存在的血检/BASDI数据
    # ==============================================
    prompt_final = f"""你是强直性脊柱炎的专科风湿免疫医师。
严格按照临床多维度交叉逻辑综合分析病情，**只基于患者已上传影像、已描述症状作答**。
严禁编造、虚构患者未提及的生化血检指标、BASDI量表评分、额外影像病变，绝对不产生病情幻觉。

回答严格分为两点：
1、病情综合分析：结合骶髂影像、临床症状，对照标准病例分层判断病情分期与活动程度
2、分层诊疗与长期预后建议：贴合指南规范用药、康复、随访方案，严谨专业、简洁通俗、不废话

参考诊疗知识库：
{context}

患者当前有效病情信息：
{full_query}
"""

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt_final}]
    }]
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        return_tensors="pt"
    ).to(model.device)

    # 优化生成参数：低温度更严谨，不发散、不乱编
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=600, 
            repetition_penalty=1.3,
            temperature=0.1,
            top_p=0.9
        )
    answer = processor.decode(out[0], skip_special_tokens=True)
    if "assistant" in answer:
        answer = answer.split("assistant")[-1]

    return answer

# =========================
# 发送逻辑（完全不动）
# =========================
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image": uploaded_image
    })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 正在分析强直性脊柱炎影像与症状，请稍等...")
        try:
            response = get_answer(prompt, uploaded_image)
        except Exception as e:
            response = f"❌ 诊断失败:\n\n{str(e)}"
        placeholder.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # 重置上传组件
    st.session_state.uploader_key += 1
    st.rerun()
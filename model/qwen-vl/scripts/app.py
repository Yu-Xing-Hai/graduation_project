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
# 推理函数（完全保留你的逻辑）
# =========================
def get_answer(text, img):
    # 从session_state中获取预加载的模型、处理器和向量数据库
    # 这些资源在应用启动时已经加载，避免重复加载
    model = st.session_state.model
    processor = st.session_state.processor
    db = st.session_state.db

    # 初始化图片描述，默认为"无图片"
    img_desc = "无图片"

    # Step1: 图片理解 - 使用视觉语言模型分析上传的医学影像
    if img:
        # 将图片转换为RGB格式，确保模型能够正确处理
        image = img.convert("RGB")
        # 设置图片理解的提示词，要求模型专业描述强直性脊柱炎影像病变
        prompt_img = """你是风湿免疫科放射影像医师，仅针对本次骶髂关节医学影像，专业描述强直性脊柱炎特征性病理病变：
1. 重点分析骶髂关节骨质侵蚀、骨质硬化、关节间隙宽窄、骨髓水肿、脊柱韧带钙化竹节样改变、肌腱附着点炎症
2. 严格使用强直性脊柱炎放射医学专业术语，简洁严谨、不冗余废话
3. 绝对禁止描述人物外貌、人脸、人像、人物情绪、图片背景等所有无关内容
4. 只输出骨骼病灶影像结论，不做额外病情推断"""
        # 构建多模态消息格式，包含图片和文本
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},  # 图片类型输入
                {"type": "text", "text": prompt_img}  # 文本提示
            ]
        }]
        # 使用processor将消息转换为模型输入格式
        # apply_chat_template: 应用聊天模板格式
        # add_generation_prompt=True: 添加生成提示
        # images=image: 传入图片数据
        # return_tensors="pt": 返回PyTorch张量
        inputs = processor(
            text=processor.apply_chat_template(messages, add_generation_prompt=True),
            images=image,
            return_tensors="pt"
        ).to(model.device)  # 将输入移动到模型所在设备（CPU/GPU）

        # 使用torch.no_grad()上下文管理器，禁用梯度计算以节省内存
        with torch.no_grad():
            # 调用模型生成图片描述
            # max_new_tokens=300: 最多生成300个新token
            # repetition_penalty=1.25: 重复惩罚，避免重复内容
            out = model.generate(**inputs, max_new_tokens=300, repetition_penalty=1.25)
        # 解码模型输出，获取文本描述
        # skip_special_tokens=True: 跳过特殊token，只保留有意义的文本
        img_desc = processor.decode(out[0], skip_special_tokens=True)
        # 清理输出，移除"assistant"前缀（如果存在）
        if "assistant" in img_desc:
            img_desc = img_desc.split("assistant")[-1]

    # Step2: RAG检索 - 使用向量数据库检索相关的专业知识
    # 构建完整查询，包含影像描述和患者症状
    full_query = f"影像：{img_desc} | 症状：{text}"  # TODO RAG下一步的添加计划
    # 使用向量数据库检索器查询相关文档
    # k=2: 返回最相关的2个文档
    docs = db.as_retriever(k=2).get_relevant_documents(full_query)
    # 将检索到的文档内容合并为上下文字符串
    context = "\n".join([d.page_content for d in docs])

    # Step3: 最终诊断 - 基于RAG检索结果生成专业诊疗建议
    # 构建最终提示词，包含角色设定、回答格式要求和参考资料
    prompt_final = f"""你是强直性脊柱炎专科医生，严格分2点回答：
1. 症状分析评估
2. 专业诊疗建议
简洁、专业、不废话

参考资料：
{context}

患者描述：
{full_query}
"""
    # 构建只包含文本的消息格式
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt_final}]
    }]
    # 使用processor将文本消息转换为模型输入格式
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        return_tensors="pt"
    ).to(model.device)

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 调用模型生成最终诊断答案
        # max_new_tokens=600: 最多生成600个新token，比图片描述更多
        # repetition_penalty=1.3: 更高的重复惩罚，确保答案多样性
        out = model.generate(**inputs, max_new_tokens=600, repetition_penalty=1.3)
    # 解码模型输出，获取最终答案
    answer = processor.decode(out[0], skip_special_tokens=True)
    # 清理输出，移除"assistant"前缀（如果存在）
    if "assistant" in answer:
        answer = answer.split("assistant")[-1]

    # 返回生成的诊断答案
    return answer

# =========================
# 发送逻辑
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
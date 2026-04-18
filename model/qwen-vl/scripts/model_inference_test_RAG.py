import warnings
import logging
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().disabled = True
logging.getLogger("transformers").disabled = True
logging.getLogger("torch").disabled = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import torch
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from config import *

# ===================== 加载VL多模态模型 =====================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model.eval()

# ===================== 加载RAG向量库（完全不动你的库） =====================
def get_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': DEVICE}
    )
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return db

# ===================== 1.VL解析图片，输出精简影像描述 =====================
def parse_image_to_text(image_path):
    image = Image.open(image_path).convert("RGB")
    prompt = "精简专业描述这张强直性骶髂关节影像病变，不要废话，简短准确"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, repetition_penalty=1.25)
    img_desc = processor.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1]
    return img_desc

# ===================== 2.RAG检索相关诊疗指南 =====================
def rag_search(db, query):
    docs = db.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    return context

# ===================== 3.统一生成规范最终报告 =====================
def multimodal_rag_predict(db, text_query=None, image_path=None):
    img_desc = ""
    # 有图片就解析
    if image_path:
        img_desc = parse_image_to_text(image_path)

    # 拼接完整查询
    full_query = f"""
    患者影像情况：{img_desc}
    患者症状描述：{text_query}
    """

    # 检索指南知识
    rag_context = rag_search(db, full_query)

    # 最终严格提示词，杜绝废话
    final_prompt = f"""
你是强直性脊柱炎专业医师，结合下方影像、患者症状、医学指南，严格分两点回答：
1、症状分析评估
2、专业诊疗建议
要求：简洁专业、分点清晰、不编造、不啰嗦、不写无关鸡汤

医学指南参考：{rag_context}
患者信息：{full_query}
"""

    messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, repetition_penalty=1.3)
    return processor.decode(out[0], skip_special_tokens=True)

# ===================== 测试 =====================
if __name__ == "__main__":
    db = get_vector_database()
    IMG = "/apps/users/icps_intelligence/data/dhc/data/AS_photos/image.png"

    print("===纯文本问答===")
    print(multimodal_rag_predict(db, text_query="强直性脊柱炎典型症状"))

    print("\n===图片+症状联合诊断===")
    print(multimodal_rag_predict(db, text_query="骶髂关节疼3个月，晨起僵硬", image_path=IMG))
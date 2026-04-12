# config.py  核心配置文件
import os
from peft import LoraConfig

# ======================== 一、下载相关 =======================
## 基础模型路径（和训练时一致）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内HuggingFace镜像加速
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 延长下载超时时间至5分钟，应对网络波动
BASE_MODEL_PATH = "/apps/users/icps_intelligence/data/dhc/model/qwen-7b-local"
# ========================= 二、LoRA相关 =======================
## LoRA权重路径（训练后保存的最终权重）
LORA_WEIGHTS_PATH = "/apps/users/icps_intelligence/data/dhc/model/lora_checkpoints/qwen-lora-as-final"
JSON_FILE_PATH = "/apps/users/icps_intelligence/data/dhc/data/LoRA/LoRA-textData.json"
## 推理设备（固定GPU 0，和训练一致）
DEVICE = "cuda:0"
## 推理参数（可根据需求调整）
INFERENCE_CONFIG = {
    "max_new_tokens": 200,    # 适配训练数据的回答长度
    "do_sample": False,       # 纯贪心解码（无随机性，最精准）
    "repetition_penalty": 1.5, # 加大重复惩罚，避免循环
    "num_beams": 1,           # 贪心解码（单beam）
}
LORA_CONFIG = LoraConfig(
    r=32,  # LoRA低秩矩阵的秩，秩越大，可训练参数越多，微调能力越强（常规8-64）
    lora_alpha=64,  # LoRA缩放因子，控制权重更新幅度，通常设置为r的2倍
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # 给Qwen模型的注意力层、输出投影层、前馈层插入LoRA
    lora_dropout=0.05,  # LoRA层的dropout率，防止训练过拟合
    bias="none",  # 不训练偏置项，最大化减少参数量
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（大模型文本生成专用）
)

# ========================== 三、RAG相关 ======================
PDF_DATA_PATH = "/apps/users/icps_intelligence/data/dhc/data/rag/pdf_guide"  # PDF医疗指南存放目录
KG_JSON_PATH = "/apps/users/icps_intelligence/data/dhc/data/rag/RAG-medicalKnowledgeGraph.json"  # 医疗知识图谱JSON文件路径
CHROMA_DB_PATH = "/apps/users/icps_intelligence/data/dhc/data/rag/chroma_db_pdf"  # PDF专属向量库保存路径（避免覆盖旧库）
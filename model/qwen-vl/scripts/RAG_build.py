import os
import re
from typing import List, Tuple, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import *

# ==========================================
# ⚙️ 配置区
# ==========================================

CHANNEL_MAP = {
    "channel_1.txt": ("综合诊疗决策通道", "comprehensive_treatment"),
    "channel_2.txt": ("影像特征与解读通道", "imaging_features"),
    "channel_3.txt": ("生化指标与临床阈值通道", "biochemistry_thresholds"),
    "channel_4.txt": ("量表评分与活动度分级通道", "assessment_scales"),
    "channel_5.txt": ("临床症状与鉴别通道", "clinical_symptoms")
}

ANATOMY_MAP = {
    "骶髂关节": "SACROILIAC_JOINT",
    "脊柱": "SPINE",
    "髋关节": "HIP_JOINT",
    "外周关节": "PERIPHERAL_JOINT",
    "足跟": "HEEL",
    "胸骨": "STERNUM",
    "骨盆": "PELVIS",
    "肋骨": "RIB",
    "肩关节": "SHOULDER_JOINT",
    "肘关节": "ELBOW_JOINT",
    "腕关节": "WRIST_JOINT",
    "膝关节": "KNEE_JOINT",
    "踝关节": "ANKLE_JOINT"
}

FINDING_MAP = {
    "炎症": "INFLAMMATION",
    "骨侵蚀": "BONE_EROSION",
    "骨硬化": "BONE_SCLEROSIS",
    "关节间隙变窄": "JOINT_SPACE_NARROWING",
    "关节强直": "JOINT_ANKYLOSIS",
    "软组织肿胀": "SOFT_TISSUE_SWELLING",
    "肌腱端炎": "ENTHESITIS",
    "新骨形成": "NEW_BONE_FORMATION"
}

def parse_channel_content(file_path: str, channel_name: str) -> List[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = re.split(r'### BLOCK', content)
    docs = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 1. 提取检索用的元数据（不修改）
        anatomy_match = re.search(r'\[解剖部位\]\s*(.+)', block)
        anatomy_text = anatomy_match.group(1).strip() if anatomy_match else "UNKNOWN"
        
        finding_match = re.search(r'\[(特征标签|指标名称|症状名称|药物类别|评分标签)\]\s*(.+)', block)
        finding_text = finding_match.group(2).strip() if finding_match else "UNKNOWN"

        anatomy_code = ANATOMY_MAP.get(anatomy_text, anatomy_text)
        finding_code = FINDING_MAP.get(finding_text, finding_text)

        # 🔥🔥🔥 修复核心：直接保存【完整BLOCK内容】，不截断、不丢失任何字段
        final_content = block

        doc = Document(
            page_content=final_content,
            metadata={
                "channel_name": channel_name,
                "anatomy": anatomy_code,
                "finding": finding_code,
                "source_block": channel_name  # 来源仅存通道名，简洁干净
            }
        )
        docs.append(doc)
    
    return docs

def build_vector_db():
    print("🚀 开始构建五通道 RAG 向量库...")
    
    print("📥 正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': DEVICE}
    )
    
    channel_dbs = {}
    
    for filename, (chinese_name, english_name) in CHANNEL_MAP.items():
        channel_path = os.path.join(CHROMA_DB_PATH, english_name)
        os.makedirs(channel_path, exist_ok=True)
        
        channel_dbs[chinese_name] = Chroma(
            collection_name=english_name,
            persist_directory=channel_path,
            embedding_function=embeddings
        )
        print(f"   ✅ 初始化通道库: {chinese_name} ({english_name})")
    
    data_dir = TXT_DATA_PATH if 'TXT_DATA_PATH' in globals() else DATA_DIR
    
    if not os.path.exists(data_dir):
        print(f"❌ 错误：数据目录不存在 -> {data_dir}")
        return

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"⚠️ 警告：在 {data_dir} 中未找到 .txt 文件")
        return

    for filename in txt_files:
        if filename not in CHANNEL_MAP:
            print(f"⚠️ 跳过未知文件: {filename}")
            continue
            
        chinese_channel_name, english_channel_name = CHANNEL_MAP[filename]
        file_path = os.path.join(data_dir, filename)
        
        print(f"\n📄 正在处理: {filename} ...")
        docs = parse_channel_content(file_path, chinese_channel_name)
        
        if not docs:
            print(f"   ❌ 该文件未提取到有效知识块，请检查格式")
            continue

        prefix = f"【{chinese_channel_name}】"
        for doc in docs:
            doc.page_content = f"{prefix} {doc.page_content}"
        
        channel_dbs[chinese_channel_name].add_documents(docs)
        print(f"   ✅ 成功写入 {len(docs)} 个知识块")

    print("\n========================================")
    print("🎉 构建完成！最终统计：")
    print("========================================")
    for chinese_name, db in channel_dbs.items():
        count = db._collection.count()
        print(f"📊 {chinese_name}: {count} 条")
    print("========================================")

if __name__ == "__main__":
    build_vector_db()
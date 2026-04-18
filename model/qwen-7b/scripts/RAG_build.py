import os
import re
import torch
import json
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.documents import Document
from peft import PeftModel
from config import *

# -------------------------- 批量加载PDF文件 --------------------------
def batch_load_pdf():
    """
    批量加载目录下所有PDF文件，自动提取文本+保留页码/文件路径元数据
    无需手动转TXT，一步到位获取LangChain可处理的Document对象
    """
    print("开始批量加载PDF医疗指南...")
    all_documents = []  # 存储所有PDF解析后的Document对象

    # 遍历PDF目录下所有.pdf文件
    for filename in os.listdir(PDF_DATA_PATH):
        if not filename.endswith(".pdf"):
            continue  # 跳过非PDF文件
        
        pdf_file_path = os.path.join(PDF_DATA_PATH, filename)
        try:
            # 核心：使用PDFPlumberLoader直接加载PDF
            loader = PDFPlumberLoader(pdf_file_path)
            # 加载PDF所有页面，返回按页拆分的Document列表
            pdf_documents = loader.load()

            # 调用统一的清洗函数（替换原有重复清洗逻辑）
            for doc in pdf_documents:
                doc.page_content = clean_text_content(doc.page_content)
            # 过滤空内容的文档
            pdf_documents = [d for d in pdf_documents if d.page_content]
            all_documents.extend(pdf_documents)
            print(f"成功加载：{filename}（共{len(pdf_documents)}页）")
        
        except FileNotFoundError:
            print(f"加载失败：{filename}，文件不存在或路径错误")
        except PermissionError:
            print(f"加载失败：{filename}，无文件读取权限")
        except Exception as e:
            print(f"加载失败：{filename}，PDF加密/损坏或其他错误：{str(e)}")
            continue

    pdf_count = len([f for f in os.listdir(PDF_DATA_PATH) if f.endswith('.pdf')])
    print(f"\nPDF加载完成汇总：共加载 {len(all_documents)} 个页面，涵盖 {pdf_count} 份PDF指南")
    return all_documents

# -------------------------- 加载医疗知识图谱JSON--------------------------
def load_medical_knowledge_graph():
    """加载JSON格式的医疗知识图谱，转换为LangChain Document对象"""
    print("\n开始加载医疗知识图谱...")
    kg_documents = []

    if not os.path.exists(KG_JSON_PATH):
        print(f"知识图谱文件不存在：{KG_JSON_PATH}")
        return kg_documents

    # 读取JSON文件
    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    # 解析每个知识图谱条目
    for idx, item in enumerate(kg_data.get("medical_knowledge_graph", [])):
        # 提取结构化字段
        scene = item.get("scene", "")
        synonyms = item.get("synonyms", [])
        node1 = item.get("node1", "")
        relation = item.get("relation", "")
        node2 = item.get("node2", "")
        evidence = item.get("evidence", "")

        # 构建自然语言文本（便于向量嵌入和检索）
        kg_text = f"""场景：{scene}
同义词：{','.join(synonyms)}
关系：{node1} {relation} {node2}
证据：{evidence}"""

        # 清洗文本
        clean_kg_text = clean_text_content(kg_text)
        if len(clean_kg_text) < 10:  # 过滤无效条目
            continue

        # 构建Document对象，保留所有结构化元数据
        kg_doc = Document(
            page_content=clean_kg_text,
            metadata={
                "source": "medical_knowledge_graph.json",
                "source_type": "knowledge_graph",
                "scene": scene,
                "synonyms": ",".join(synonyms),
                "node1": node1,
                "relation": relation,
                "node2": node2,
                "evidence": evidence,
                "kg_id": idx  # 知识图谱条目唯一标识
            }
        )
        kg_documents.append(kg_doc)

    print(f"知识图谱加载完成：共转换 {len(kg_documents)} 个结构化条目")
    return kg_documents

# -------------------------- 文本分割 --------------------------
def split_documents(documents):
    print("\n开始分割PDF文本片段（优化分片，提升检索精准度）...")
    # 文本预处理：过滤过短的无效片段
    cleaned_documents = []
    for doc in documents:
        if len(doc.page_content) > 50:
            cleaned_documents.append(doc)
    
    # 文本分割参数（旧版）
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=400,  # 每个文本片段的字符数（适中，便于模型提取关键信息）
    #     chunk_overlap=50,  # 片段间重叠字符数（避免信息断裂，提升上下文连贯性）
    #     length_function=len,  # 字符长度计算方式
    #     separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]  # 按中文标点分割，更贴合中文文本
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # 扩大到800，保留完整医学知识点
        chunk_overlap=150,  # 增加重叠，防止知识点断裂
        separators=["\n\n", "\n", "。", "；", "，", " "]  # 简化标点，更稳定
    )
    
    # 执行文本分割
    splits = text_splitter.split_documents(cleaned_documents)
    print(f"✅ 文本分割完成：共得到 {len(splits)} 个有效文本片段（已清洗乱码和冗余内容）")
    return splits

# -------------------------- 通用文本清洗工具函数--------------------------
# 文本清洗工具函数（旧版）
# def clean_text_content(raw_text):
#     """
#     通用文本清洗函数（统一复用，避免多处重复逻辑）
#     保留英文字母和医疗缩写
#     """
#     if not raw_text:
#         return ""
#     # re.sub(匹配规则, 替换成什么, 处理的文本)
#     clean_text = re.sub(r'\n+', '\n', raw_text)
#     clean_text = re.sub(r'\s+', ' ', clean_text)
#     clean_text = re.sub(r'[a-zA-Z0-9_]{10,}', '', clean_text)  # 删除长串乱码
#     clean_text = re.sub(r"[^\u4e00-\u9fff\w\.\,\;\:\!\?\(\)\-\n]", " ", clean_text)
#     clean_text = re.sub(r'\n+', ' ', clean_text)
#     clean_text = re.sub(r'\s+', ' ', clean_text).strip()
#     return clean_text

def clean_text_content(raw_text):
    """
    温和清洗：保留医疗术语、标点、符号，只删乱码/冗余空格
    """
    if not raw_text:
        return ""
    # 1. 清理PDF页码、特殊符号
    text = re.sub(r'第\d+页|共\d+页|Page \d+|[\uf0b7\uf0fc]', '', raw_text)
    # 2. 统一空格/空行
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # 3. 仅删除无效特殊字符，保留医疗常用符号 - / ( ) [ ]
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s\.\,\;\:\(\)\[\]\-\/]", " ", text)
    return text.strip()

# -------------------------- 构建Chroma向量数据库 --------------------------
def build_vector_database(document_splits):
    """将分割后的文本片段转为向量并构建持久化向量库"""
    print("\n开始构建/加载向量数据库（持久化存储）...")
    # text2vec-base-chinese：中文通用最优轻量模型，支持医疗术语表征
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': DEVICE}
    )
    
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        print(f"向量库已存在，直接加载：{CHROMA_DB_PATH}")
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print(f"向量库不存在，开始构建：{CHROMA_DB_PATH}")
        db = Chroma.from_documents(
            documents=document_splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
    
    print(f"向量数据库处理完成，已保存/加载至目录：{CHROMA_DB_PATH}")
    return db

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 步骤1：加载PDF文档
        pdf_documents = batch_load_pdf()

        # 步骤2：加载知识图谱文档
        kg_documents = load_medical_knowledge_graph()

        # 步骤3：分割文本片段
        document_splits = split_documents(pdf_documents)

        # 步骤4：合并所有文档（PDF+知识图谱）
        all_documents = document_splits + kg_documents
        print(f"\n合并文档完成：PDF({len(document_splits)}) + 知识图谱({len(kg_documents)}) = 总计{len(all_documents)}个文档")

        # 步骤5：构建混合向量数据库
        chroma_db = build_vector_database(all_documents)
        
        print("\n全流程运行完成！PDF+知识图谱驱动的RAG医疗知识库已搭建成功。")
    except Exception as e:
        import traceback
        print(f"\n运行出错：{str(e)}")
        print(f"异常详情：\n{traceback.format_exc()}")
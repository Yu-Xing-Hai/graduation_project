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

# -------------------------- 1. 全局路径配置（适配你的项目结构）--------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内HuggingFace镜像加速
LOCAL_MODEL_PATH = "../qwen-7b-local"  # 本地Qwen-7B模型路径
PDF_DATA_PATH = "../../data/rag/pdf_guide/"  # PDF医疗指南存放目录
KG_JSON_PATH = "../../data/rag/RAG-medicalKnowledgeGraph.json"  # 医疗知识图谱JSON文件路径
CHROMA_DB_PATH = "../../data/rag/chroma_db_pdf"  # PDF专属向量库保存路径（避免覆盖旧库）

# -------------------------- 2. 核心：直接批量加载PDF文件（无需转TXT）--------------------------
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

            # 优化：调用统一的清洗函数（替换原有重复清洗逻辑）
            for doc in pdf_documents:
                doc.page_content = clean_text_content(doc.page_content)
            
            # 合并当前PDF的所有页面文档到总列表
            all_documents.extend(pdf_documents)
            print(f"成功加载：{filename}（共{len(pdf_documents)}页）")
        
        # 优化：细化异常捕获，精准排查问题
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

# -------------------------- 3. 新增：加载医疗知识图谱JSON--------------------------
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

# -------------------------- 3. 文本分割（适配Qwen-7B上下文，保证连贯性）--------------------------
def split_documents(documents):
    print("\n开始分割PDF文本片段（优化分片，提升检索精准度）...")
    # 1. 文本预处理：过滤过短的无效片段
    cleaned_documents = []
    for doc in documents:
        if len(doc.page_content) > 50:
            cleaned_documents.append(doc)
    
    # 2. 配置文本分割参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # 每个文本片段的字符数（适中，便于模型提取关键信息）
        chunk_overlap=50,  # 片段间重叠字符数（避免信息断裂，提升上下文连贯性）
        length_function=len,  # 字符长度计算方式
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]  # 按中文标点分割，更贴合中文文本
    )
    
    # 3. 执行文本分割
    splits = text_splitter.split_documents(cleaned_documents)
    print(f"✅ 文本分割完成：共得到 {len(splits)} 个有效文本片段（已清洗乱码和冗余内容）")
    return splits

# -------------------------- 新增：通用文本清洗工具函数（复用逻辑，避免冗余）--------------------------
def clean_text_content(raw_text):
    """
    通用文本清洗函数（统一复用，避免多处重复逻辑）
    保留英文字母和医疗缩写
    """
    if not raw_text:
        return ""
    # re.sub(匹配规则, 替换成什么, 处理的文本)
    clean_text = re.sub(r'\n+', '\n', raw_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[a-zA-Z0-9_]{10,}', '', clean_text)  # 删除长串乱码
    clean_text = re.sub(r"[^\u4e00-\u9fff\w\.\,\;\:\!\?\(\)\-\n]", " ", clean_text)
    clean_text = re.sub(r'\n+', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# -------------------------- 4. 构建Chroma向量数据库（持久化存储，后续可复用）--------------------------
def build_vector_database(document_splits):
    """将分割后的文本片段转为向量并构建持久化向量库"""
    print("\n开始构建/加载向量数据库（持久化存储）...")
    # text2vec-base-chinese：中文通用最优轻量模型，支持医疗术语表征
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cuda'}
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

# -------------------------- 5. 加载本地Qwen-7B大模型（适配医疗问答）--------------------------
def load_local_qwen_7b():
    """加载本地Qwen-7B模型，封装为LangChain可调用的LLM对象"""
    print("\n开始加载本地Qwen-7B模型...")
    # 加载tokenizer分词器
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )

    # 优化：增加设备判断，自动适配数据类型（兼容CPU/GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"检测到运行设备：{device}，使用数据类型：{torch_dtype}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch_dtype,    # 自动适配设备的数据类型
        device_map="auto",          # 自动分配设备（有GPU用GPU，无GPU用CPU）
        low_cpu_mem_usage=True      # 优化CPU内存占用
    )

    # 封装为HuggingFace Pipeline，适配LangChain
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        top_p=0.85,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # 转换为LangChain LLM对象
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    print("本地Qwen-7B模型加载完成，可用于RAG问答")
    return llm

# -------------------------- 6. 搭建RAG检索增强问答链--------------------------
def build_rag_qa_chain(db, llm):
    print("\n开始搭建RAG检索增强问答链（融合PDF+知识图谱）...")
    # 最终版Prompt：强制格式+禁止额外内容
    prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""严格遵守以下规则回答：
1. 仅输出【核心结论】和【分点依据】，无任何额外文字、列表、解释、问候语；
2. 核心结论≤50字，分点依据仅写2条，每条必须标注来源；
3. 仅使用医疗知识库中的信息，不编造未提及的指南/手册。

核心结论：一句话总结
分点依据：
1. 具体依据（标注来源：PDF指南/知识图谱 + 证据信息）
2. 具体依据（标注来源：PDF指南/知识图谱 + 证据信息）

### 医疗知识库
{context}

### 用户问题
{question}

### 专业回答""",
)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    print("RAG问答链搭建完成（融合PDF+知识图谱）")
    return qa_chain

# -------------------------- 7. 测试RAG效果（医疗场景针对性验证）--------------------------
def test_rag_qa(qa_chain):
    print("\n=====================================")
    print("开始测试PDF+知识图谱驱动的RAG医疗知识库")
    print("=====================================\n")

    test_questions = [
        "中轴型脊柱关节炎（axSpA）包含哪些子集？",  # 知识图谱核心问题
        "ASDAS评分在强直性脊柱炎诊疗中的作用是什么？",  # 混合PDF+知识图谱
        "强直性脊柱炎的中医证型有哪些？",  # 知识图谱中医相关
    ]

    for idx, question in enumerate(test_questions, 1):
        print(f"【测试问题 {idx}】")
        print(f"问题：{question}\n")
        print("-" * 60 + "\n")

        result = qa_chain.invoke({"query": question})

        # 修复核心：容错访问元数据，避免KeyError
        unique_source_docs = []
        seen_identifiers = set()
        for doc in result["source_documents"]:
            source_type = doc.metadata.get("source_type", "unknown")
            # 分类型生成唯一标识，全部用get方法+默认值
            if source_type == "pdf":
                doc_file = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
                doc_page = doc.metadata.get("page", 0)
                doc_identifier = (source_type, doc_file, doc_page)
            else:  # 知识图谱
                doc_source = doc.metadata.get("source", "unknown_kg.json")
                doc_kg_id = doc.metadata.get("kg_id", -1)  # 默认值-1，避免KeyError
                doc_identifier = (source_type, doc_source, doc_kg_id)
            # 内容哈希去重
            doc_hash = hash(doc.page_content.strip())
            full_identifier = (doc_identifier, doc_hash)
            
            if full_identifier not in seen_identifiers:
                seen_identifiers.add(full_identifier)
                unique_source_docs.append(doc)

        # 打印回答
        print(f"【专业回答】\n{result['result'].strip()}\n")

        # 打印来源（区分PDF和知识图谱，同样容错）
        print(f"【参考来源】")
        if len(unique_source_docs) > 0:
            for src_idx, source_doc in enumerate(unique_source_docs[:2]):
                source_type = source_doc.metadata.get("source_type", "unknown")
                if source_type == "pdf":
                    source_file = os.path.basename(source_doc.metadata.get("source", "unknown.pdf"))
                    page_num = source_doc.metadata.get("page", 0) + 1
                    source_info = f"{source_file} 第{page_num}页"
                else:
                    scene = source_doc.metadata.get("scene", "未知场景")
                    evidence = source_doc.metadata.get("evidence", "")[:30]  # 只取前30字
                    source_info = f"医疗知识图谱（场景：{scene}，证据：{evidence}...）"
                print(f"  {src_idx+1}. {source_info}")
        else:
            print("  无有效参考来源")
        
        print("\n" + "=" * 60 + "\n")

# -------------------------- 主函数：串联全流程（一键运行）--------------------------
if __name__ == "__main__":
    try:
        # 步骤1：加载PDF文档
        pdf_documents = batch_load_pdf()

        # 步骤2：加载知识图谱文档
        kg_documents = load_medical_knowledge_graph()

        # 步骤3：合并所有文档（PDF+知识图谱）
        all_documents = pdf_documents + kg_documents
        print(f"\n合并文档完成：PDF({len(pdf_documents)}) + 知识图谱({len(kg_documents)}) = 总计{len(all_documents)}个文档")

        # 步骤4：分割文本片段
        document_splits = split_documents(all_documents)

        # 步骤5：构建混合向量数据库
        chroma_db = build_vector_database(document_splits)

        # 步骤6：加载本地Qwen-7B模型
        qwen_llm = load_local_qwen_7b()

        # 步骤7：搭建RAG问答链
        rag_qa_chain = build_rag_qa_chain(chroma_db, qwen_llm)

        # 步骤8：测试RAG效果
        test_rag_qa(rag_qa_chain)

        print("\n全流程运行完成！PDF+知识图谱驱动的RAG医疗知识库已搭建成功。")
    except Exception as e:
        import traceback
        print(f"\n运行出错：{str(e)}")
        print(f"异常详情：\n{traceback.format_exc()}")
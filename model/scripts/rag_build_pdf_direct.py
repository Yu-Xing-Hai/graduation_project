import os
import re
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# -------------------------- 1. 全局路径配置（适配你的项目结构）--------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内HuggingFace镜像加速
LOCAL_MODEL_PATH = "../qwen-7b-local"  # 本地Qwen-7B模型路径
PDF_DATA_PATH = "../../data/rag/pdf_guide/"  # PDF医疗指南存放目录
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
        chunk_size=500,  # 每个文本片段的字符数（适中，便于模型提取关键信息）
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
    print("\n开始搭建RAG检索增强问答链...")
    # 核心修改：无多余缩进+强制内容填充+禁止多余序号
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""你是专业风湿免疫科医生，仅基于医疗指南知识库回答问题，严格遵守：
1. 只提取有效信息，严禁生成空序号、多余序号（仅保留2条分点），严禁复述指令/知识库；
2. 无相关信息仅回复「未在医疗指南中查询到相关内容」，有信息则按以下格式填充具体内容：
核心结论：一句话总结（≤50字）
分点依据：
1. 具体依据（标注指南名称+页码）
2. 具体依据（标注指南名称+页码）

### 医疗指南知识库
{context}

### 用户问题
{question}

### 专业回答""",
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # 新增相似度阈值，过滤乱码/无关片段
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    print("RAG检索增强问答链搭建完成")
    return qa_chain

# -------------------------- 7. 测试RAG效果（医疗场景针对性验证）--------------------------
def test_rag_qa(qa_chain):
    print("\n=====================================")
    print("开始测试PDF驱动的RAG医疗知识库")
    print("=====================================\n")

    test_questions = [
        "强直性脊柱炎合并炎症性肠病（IBD），使用TNFi治疗无效时，推荐使用什么药物？",
        "ASDAS评分在强直性脊柱炎诊疗中的作用是什么？",
    ]

    for idx, question in enumerate(test_questions, 1):
        print(f"【测试问题 {idx}】")
        print(f"问题：{question}\n")
        print("-" * 60 + "\n")

        result = qa_chain.invoke({"query": question})

        unique_source_docs = []
        seen_identifiers = set()
        for doc in result["source_documents"]:  # 直接使用原始检索结果，不做额外过滤
            doc_file = os.path.basename(doc.metadata["source"])
            doc_page = doc.metadata["page"]
            doc_hash = hash(doc.page_content.strip())
            doc_identifier = (doc_file, doc_page, doc_hash)
            if doc_identifier not in seen_identifiers:
                seen_identifiers.add(doc_identifier)
                unique_source_docs.append(doc)

        print(f"【专业回答】\n{result['result'].strip()}\n")

        print(f"【参考指南来源】")
        if len(unique_source_docs) > 0:
            for src_idx, source_doc in enumerate(unique_source_docs[:2]):
                source_file = os.path.basename(source_doc.metadata["source"])
                page_num = source_doc.metadata["page"] + 1
                print(f"  {src_idx+1}. {source_file} 第{page_num}页")
        else:
            print("  无有效参考来源")
        
        print("\n" + "=" * 60 + "\n")

# -------------------------- 主函数：串联全流程（一键运行）--------------------------
if __name__ == "__main__":
    try:
        # 步骤1：直接加载PDF（无需转TXT）
        pdf_documents = batch_load_pdf()

        # 步骤2：分割文本片段
        document_splits = split_documents(pdf_documents)

        # 步骤3：构建向量数据库
        chroma_db = build_vector_database(document_splits)

        # 步骤4：加载本地Qwen-7B模型
        qwen_llm = load_local_qwen_7b()

        # 步骤5：搭建RAG问答链
        rag_qa_chain = build_rag_qa_chain(chroma_db, qwen_llm)

        # 步骤6：测试RAG效果
        test_rag_qa(rag_qa_chain)

        print("\n全流程运行完成！PDF驱动的RAG医疗知识库已搭建成功。")
    except Exception as e:
        # 优化：打印异常详情（包括行号），方便排查
        import traceback
        print(f"\n运行出错：{str(e)}")
        print(f"异常详情：\n{traceback.format_exc()}")
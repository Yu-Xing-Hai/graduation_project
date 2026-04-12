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

import os
from config import CHROMA_DB_PATH, DEVICE
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_vector_database():
    """
    加载本地持久化的Chroma向量数据库
    若向量库不存在：抛出明确异常，禁止返回None
    若向量库存在：直接加载并返回数据库对象
    """
    print("\n开始构建/加载向量数据库（持久化存储）...")
    
    # 初始化向量嵌入模型（使用配置的设备，不硬编码）
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': DEVICE}  # 用config的设备，自动适配CPU/GPU
    )
    
    # 校验向量库是否存在
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        print(f"✅ 向量库已存在，直接加载：{CHROMA_DB_PATH}")
        try:
            db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            return db
        except Exception as e:
            raise RuntimeError(f"向量库加载失败：{str(e)}")
    else:
        # 关键：不存在时直接抛错，不返回None！
        raise FileNotFoundError(
            f"❌ 向量库不存在！请先运行向量库构建脚本生成数据\n路径：{CHROMA_DB_PATH}"
        )

# -------------------------- 5. 加载本地Qwen-7B大模型（适配医疗问答）--------------------------
def load_local_qwen_7b():
    """加载本地Qwen-7B模型，封装为LangChain可调用的LLM对象"""
    print("\n开始加载本地Qwen-7B模型...")
    # 加载tokenizer分词器
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )

    # 优化：增加设备判断，自动适配数据类型（兼容CPU/GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"检测到运行设备：{device}，使用数据类型：{torch_dtype}")

    # 加载模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch_dtype,    # 自动适配设备的数据类型
        device_map="auto",          # 自动分配设备（有GPU用GPU，无GPU用CPU）
        low_cpu_mem_usage=True      # 优化CPU内存占用
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_WEIGHTS_PATH,  # LoRA权重路径
        device_map="auto"
    )

    lora_model.eval()  # 推理模式
    print(f"✅ LoRA权重加载完成：{LORA_WEIGHTS_PATH}")

    # 封装为HuggingFace Pipeline，适配LangChain
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=lora_model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.0,     # 纯贪心解码，和LoRA推理一致
        do_sample=False,     # 关闭采样，消除top_p/top_k警告
        top_p=1.0,           # 无效参数置为默认值
        top_k=0,
        repetition_penalty=1.5,
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
        # 步骤5：加载混合向量数据库
        chroma_db = get_vector_database()

        # 步骤6：加载本地Qwen-7B模型
        qwen_llm = load_local_qwen_7b()

        # 步骤7：搭建RAG问答链
        rag_qa_chain = build_rag_qa_chain(chroma_db, qwen_llm)

        # 步骤8：测试RAG效果
        test_rag_qa(rag_qa_chain)

        print("\nLoRA-RAG模型推理测试完成！")
    except Exception as e:
        import traceback
        print(f"\n运行出错：{str(e)}")
        print(f"异常详情：\n{traceback.format_exc()}")
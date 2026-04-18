import warnings
import logging
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch.utils._pytree._register_pytree_node.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().disabled = True
logging.getLogger("transformers").disabled = True
logging.getLogger("torch").disabled = True
logging.getLogger("huggingface_hub").disabled = True
logging.getLogger("peft").disabled = True
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_SUPPRESS_LOGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

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
import sys

def get_vector_database():
    """加载本地持久化Chroma向量数据库"""
    print("\n开始构建/加载向量数据库（持久化存储）...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': DEVICE}
    )
    
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
        raise FileNotFoundError(
            f"❌ 向量库不存在！请先运行向量库构建脚本生成数据\n路径：{CHROMA_DB_PATH}"
        )

def load_local_qwen_lora():
    """加载Qwen-7B+LoRA，无警告、稳定推理"""
    print("\n开始加载本地Qwen-7B模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"检测到运行设备：{device}，使用数据类型：{torch_dtype}")

    # 加载基础模型+LoRA参数
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True
    )
    lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH, device_map="auto")
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=lora_model,
        tokenizer=tokenizer,
        max_new_tokens=256,    # 缩短长度，禁止啰嗦
        do_sample=False,       # 确定性生成
        repetition_penalty=1.3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    print("本地Qwen-7B模型加载完成，可用于RAG问答")
    return llm

def build_rag_qa_chain(db, llm):
    print("\n开始搭建RAG检索增强问答链（融合PDF+知识图谱）...")
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""你是强直性脊柱炎(AS)专业诊疗助手，**只根据下方知识库回答**，禁止编造、禁止输出知识库以外的内容。
回答要求：
1. 问什么答什么，不强行输出风险等级、判定依据
2. 语言简洁专业，分点清晰，不带任何「场景/同义词/关系」等标签
3. 不添加多余文字、符号、注释，只输出有效诊疗信息
4. 若问题是判断类（是不是/有没有），先明确给出结论，再简要说明

知识库：
{context}

用户问题：{question}
专业回答：
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    print("RAG问答链搭建完成（融合PDF+知识图谱）")
    return qa_chain

def is_as_related(question: str) -> bool:
    """毕设最稳方案：关键词匹配，覆盖专业术语+普通人口语"""
    as_keywords = [
        # 专业词
        "强直性脊柱炎", "AS", "骶髂", "脊柱", "晨僵",
        # 日常口语
        "腰疼", "背疼", "脊椎", "背僵", "腰不舒服", "关节僵硬"
    ]
    return any(keyword in question for keyword in as_keywords)

# -------------------------- 测试函数 --------------------------
def test_rag_qa(qa_chain):
    print("\n=====================================")
    print("强直性脊柱炎智能诊疗系统 - LoRA+RAG推理测试")
    print("=====================================\n")

    test_questions = [
        "我早上起来背僵，活动一会就好，是强直性脊柱炎吗？",
        "强直性脊柱炎的典型症状有哪些？",
        "40岁男性，骶髂关节双侧Ⅱ级损伤，风险等级是多少？",
        "强直性脊柱炎的中医证型和干预方案是什么？",
        "我腰疼好几个月了，怎么办？",
        "感冒了应该吃什么药？",
        "膝盖疼是不是AS的症状？"
    ]

    for idx, question in enumerate(test_questions, 1):
        print(f"【测试问题 {idx}】{question}")
        print("-" * 50)

        if is_as_related(question):
            result = qa_chain.invoke({"query": question})
            answer = result['result'].strip()
            # 清理多余空行/乱码
            answer = re.sub(r'\n+', '\n', answer)
            try:
                source = result['source_documents'][0].metadata.get('source_type','医疗知识库')
                source_info = f"【参考来源】{source}"
            except:
                source_info = "【参考来源】无"
        else:
            answer = "我专注于强直性脊柱炎相关咨询~ 身体不适建议多休息、多喝水，持续不适请及时就医。"
            source_info = "【参考来源】健康通识建议"

        print(f"【规范诊疗回答】\n{answer}\n")
        print(f"{source_info}\n")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        chroma_db = get_vector_database()
        qwen_llm = load_local_qwen_lora()
        rag_qa_chain = build_rag_qa_chain(chroma_db, qwen_llm)
        test_rag_qa(rag_qa_chain)
        print("\n✅ LoRA+RAG 推理测试完成！")
    except Exception as e:
        import traceback
        print(f"\n运行出错：{str(e)}")
        print(traceback.format_exc())